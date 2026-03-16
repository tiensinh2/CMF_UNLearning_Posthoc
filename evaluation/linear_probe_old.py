import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import torch.nn.functional as F

def pool_tensor(tensor, mode='avg'):
    """
    If tensor is [B,C,H,W], immediately reduces it to [B,C] using a single
    C‐optimized call.  Otherwise returns it unchanged.
    """
    if tensor.dim() == 4:
        if mode == 'avg':
            # adaptive_avg_pool2d will give you a [B,C,1,1] tensor
            # which you can then squeeze to [B,C].
            out = F.adaptive_avg_pool2d(tensor, 1)
        elif mode == 'max':
            out = F.adaptive_max_pool2d(tensor, 1)
        elif mode == 'flatten':
            out = tensor   
        else:
            raise ValueError(f"Unknown feat_mode: {mode}")
        return out.view(out.size(0), -1)
    return tensor


def register_hooks_resnet(model):
    acts, hooks = {}, []
    acts['conv1'] = []
    hooks.append(model.conv1.register_forward_hook(lambda m, i, o: acts['conv1'].append(o.detach().cpu())))
    for lname in ['layer1','layer2','layer3','layer4']:
        layer = getattr(model, lname)
        for idx, block in enumerate(layer):
            key = f"{lname}.{idx}.bn2"
            acts[key] = []
            hooks.append(block.bn2.register_forward_hook(lambda _m,_i,o,name=key: acts[name].append(o.detach().cpu())))
    acts['fc'] = []
    hooks.append(model.fc.register_forward_hook(lambda m, i, o: acts['fc'].append(o.detach().cpu())))
    return acts, hooks


import torch.nn as nn

def get_probe_layer_names(model):
    """
    Return a list of module–names in the order they appear in the forward pass,
    but only those that are Conv2d or Linear.
    """
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
    return names


def register_hooks(model, arch):
    """
    Walk model.named_modules(), and for every nn.Conv2d or nn.Linear,
    register a forward‐hook that saves its output under acts[name].
    Works for both ResNet and your VGG above.
    """
    acts, handles = OrderedDict(), []
    name2mod = dict(model.named_modules())
    arch_low = arch.lower()

    def make_hook(name):
        def hook(m, inp, out):
            # pool_tensor comes from your code above
            pooled = pool_tensor(out, mode='avg')
            acts[name].append(pooled.detach().cpu())
        return hook

    print("Linear prob:")
    if 'resnet18' in arch_low:

        # ResNet: hook every Conv2d + the final FC
        for name, module in name2mod.items():
            #if name == 'conv1' or "bn2" in name or name == 'fc':
                print(name)
                acts[name] = []
                handles.append(module.register_forward_hook(make_hook(name)))

    elif 'resnet50' in arch_low:
        # ResNet: hook every Conv2d + the final FC
        for name, module in name2mod.items():
            if name == 'conv1' or "bn3" in name or name == 'fc':
                print(name)
                acts[name] = []
                handles.append(module.register_forward_hook(make_hook(name)))

    elif 'vgg' in arch_low:
        # VGG: only even‐indexed feature conv layers
        for idx, module in enumerate(model.features):
            if  isinstance(module, nn.Conv2d):
                name = f'features.{idx}'
                print(name)
                acts[name] = []
                handles.append(module.register_forward_hook(make_hook(name)))
        # and only even‐indexed classifier linear layers
        for idx, module in enumerate(model.classifier):
            if  isinstance(module, nn.Linear):
                name = f'classifier.{idx}'
                print(name)
                acts[name] = []
                handles.append(module.register_forward_hook(make_hook(name)))

    else:
        raise ValueError(f"Unsupported arch for hook registration: {arch}")

    return acts, handles

def clear_hooks(hooks):
    for h in hooks:
        h.remove()


def collect_features(model, loader, device, acts):
    """
    Extract and pool intermediate activations for all layers in 'acts'.
    - Clears previous buffers.
    - Runs model in eval mode under torch.no_grad().
    - Moves features to CPU before pooling and numpy conversion.
    """
    # 1) Reset buffers
    for buf in acts.values():
        buf.clear()

    # 2) Feature collection
    model.eval()
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _ = model(x)
            # collect labels
            labels_list.append(y.cpu().numpy())

    # 3) Pool and convert features
    feats_buf = {}
    for layer, buf in acts.items():
        # buf contains list of tensors [batch, C, H, W]
        t = torch.cat(buf, dim=0).cpu()              # [N, C, H, W] on CPU
        pooled = pool_tensor(t, 'avg')               # [N, C] still CPU
        feats_buf[layer] = pooled.numpy()

    # 4) Concatenate labels
    labels = np.concatenate(labels_list, axis=0)
    return feats_buf, labels


def linear_probe(args, model, train_loader, test_loader,
                 num_classes, bs_probe, device='cpu'):
    """
    Linear-probe per layer on CPU, converting one layer at a time.
    - Avoids loading all layers' tensors to memory simultaneously.
    - torch.from_numpy returns CPU tensor, separate from GPU.
    """
    # 1) extract all features (still numpy on CPU)
    model.eval()
    acts, hooks = register_hooks(model, args.arch)
    with torch.no_grad():
        feat_tr, lab_tr = collect_features(model, train_loader, device, acts)
        feat_te, lab_te = collect_features(model, test_loader,  device, acts)
    clear_hooks(hooks)

    # prepare constant label tensors once
    labels_tr = torch.tensor(lab_tr, dtype=torch.long)
    labels_te = torch.tensor(lab_te, dtype=torch.long)
    forget_cls = torch.tensor(args.unlearn_class)
    def make_masks(y):
        is_forget = (y.unsqueeze(0)==forget_cls.unsqueeze(1)).any(0)
        return ~is_forget, is_forget

    results = []
    # 2) iterate per layer, convert only this layer to tensors
    for layer in feat_tr.keys():
        # convert numpy to CPU tensor
        Xtr = torch.from_numpy(feat_tr[layer]).float()
        Xte = torch.from_numpy(feat_te[layer]).float()
        retain_tr, forget_tr = make_masks(labels_tr)
        retain_te, forget_te = make_masks(labels_te)

        C = Xtr.size(1)
        torch.manual_seed(args.seed)
        clf = nn.Linear(C, num_classes).to(device)
        opt = optim.SGD(clf.parameters(), lr=0.05, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        # DataLoader on CPU tensors
        ds = TensorDataset(Xtr, labels_tr)
        probe_loader = DataLoader(ds, batch_size=bs_probe,
                                  shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        
        # early stopping setup
        best_acc = 0.0
        patience = 30
        no_improve = 0
        max_epochs = 100
        
        # train
        for _ in range(max_epochs):
            clf.train()
            for bx, by in probe_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss_fn(clf(bx), by).backward()
                opt.step()
            
            # evaluate overall train accuracy
            clf.eval()
            with torch.no_grad():
                preds = clf(Xtr.to(device)).argmax(dim=1)
            acc_all = (preds == labels_tr).float().mean().item()

            if acc_all > best_acc:
                best_acc = acc_all
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        # Final evaluation on both train and test
        clf.eval()
        with torch.no_grad():
            pr_tr = clf(torch.from_numpy(Xtr).float().to(device)).argmax(1).cpu().numpy()
            pr_te = clf(torch.from_numpy(Xte).float().to(device)).argmax(1).cpu().numpy()

        fm_tr = np.isin(lab_tr, args.unlearn_class)
        rm_tr = ~fm_tr
        fm_te = np.isin(lab_te, args.unlearn_class)
        rm_te = ~fm_te

        results.append({
            'layer': layer,
            'acc_train_retain': float((pr_tr[rm_tr] == lab_tr[rm_tr]).mean()) if rm_tr.any() else None,
            'acc_train_forget': float((pr_tr[fm_tr] == lab_tr[fm_tr]).mean()) if fm_tr.any() else None,
            'acc_test_retain': float((pr_te[rm_te] == lab_te[rm_te]).mean()) if rm_te.any() else None,
            'acc_test_forget': float((pr_te[fm_te] == lab_te[fm_te]).mean()) if fm_te.any() else None,
            'best_train_acc': best_acc
        })

    return results

def linear_probe_CMF(args, model, train_loader, test_loader,
                             num_classes, bs_probe, device='cpu'):
    """
    Only register hook to layer4.1.bn2 of resnet18，
    Collect features from this layer after pooling for linear probing。
    """
    # 1) Register hook to layer4.1.bn2
    acts = {'layer4.1.bn2': [], 'fc': []}
    handle = model.encoder.layer4[1].bn2.register_forward_hook(
        lambda m, inp, out: acts['layer4.1.bn2'].append(
            pool_tensor(out, mode='avg').detach().cpu()
        )
    )
    hooks = [handle]
    handle_fc = model.encoder.fc.register_forward_hook(
        lambda m, inp, out: acts['fc'].append(
            pool_tensor(out, mode='avg').detach().cpu()
        )
    )
    hooks.append(handle_fc)

    with torch.no_grad():
        feat_tr, lab_tr = collect_features(model, train_loader, device, acts)
        feat_te, lab_te = collect_features(model, test_loader,  device, acts)
    clear_hooks(hooks)

    # prepare constant label tensors once
    labels_tr = torch.tensor(lab_tr, dtype=torch.long)
    labels_te = torch.tensor(lab_te, dtype=torch.long)
    forget_cls = torch.tensor(args.unlearn_class)
    def make_masks(y):
        is_forget = (y.unsqueeze(0)==forget_cls.unsqueeze(1)).any(0)
        return ~is_forget, is_forget

    results = []
    # 2) iterate per layer, convert only this layer to tensors
    for layer in feat_tr.keys():
        # convert numpy to CPU tensor
        Xtr = torch.from_numpy(feat_tr[layer]).float()
        Xte = torch.from_numpy(feat_te[layer]).float()
        retain_tr, forget_tr = make_masks(labels_tr)
        retain_te, forget_te = make_masks(labels_te)

        C = Xtr.size(1)
        torch.manual_seed(args.seed)
        clf = nn.Linear(C, num_classes).to(device)
        opt = optim.SGD(clf.parameters(), lr=0.05, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        # DataLoader on CPU tensors
        ds = TensorDataset(Xtr, labels_tr)
        probe_loader = DataLoader(ds, batch_size=bs_probe,
                                  shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        
        # early stopping setup
        best_acc = 0.0
        patience = 10
        no_improve = 0
        max_epochs = 100
        
        # train
        for epoch in range(max_epochs):
            clf.train()
            for bx, by in probe_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss_fn(clf(bx), by).backward()
                opt.step()
            
            # evaluate overall train accuracy
            clf.eval()
            with torch.no_grad():
                preds = clf(Xtr.to(device)).argmax(dim=1)
            acc_all = (preds == labels_tr.to(device)).float().mean().item()

            if acc_all > best_acc:
                best_acc = acc_all
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        # Final evaluation on both train and test
        clf.eval()
        with torch.no_grad():
            pr_tr = clf(Xtr.float().to(device)).argmax(1).cpu().numpy()
            pr_te = clf(Xte.float().to(device)).argmax(1).cpu().numpy()

        fm_tr = np.isin(lab_tr, args.unlearn_class)
        rm_tr = ~fm_tr
        fm_te = np.isin(lab_te, args.unlearn_class)
        rm_te = ~fm_te

        results.append({
            'layer': layer,
            'acc_train_retain': float((pr_tr[rm_tr] == lab_tr[rm_tr]).mean()) if rm_tr.any() else None,
            'acc_train_forget': float((pr_tr[fm_tr] == lab_tr[fm_tr]).mean()) if fm_tr.any() else None,
            'acc_test_retain': float((pr_te[rm_te] == lab_te[rm_te]).mean()) if rm_te.any() else None,
            'acc_test_forget': float((pr_te[fm_te] == lab_te[fm_te]).mean()) if fm_te.any() else None,
            'best_train_acc': best_acc,
            'epochs_trained':   epoch + 1
        })

    return results


import contextlib, torch, random, numpy as np

@contextlib.contextmanager
def rng_sandbox():
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def run_linear_probe_on_fresh_clone(
    args, get_model_fn, device_probe, src_model,
    train_loader, test_loader, num_classes, bs_probe
):
    
    state = src_model.state_dict()
    probe_model = get_model_fn(args, device_probe)  # get_model(args, device)
    print("New probe model created.")
    probe_model.load_state_dict(state, strict=True)
    print("State dict loaded into probe model.")

    probe_model.eval()
    for p in probe_model.parameters():
        p.requires_grad_(False)
        
    if "CMF" in args.unlearn_method:
        with  rng_sandbox():
            outs_LP = linear_probe_CMF_RemoveFC(
                args=args,
                model=probe_model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                bs_probe=bs_probe,
                device=str(device_probe)  # for example "cpu" or "cuda:1"
            )

        if isinstance(outs_LP, list):
            if len(outs_LP) == 0:
                raise RuntimeError("linear_probe_CMF_RemoveFC return an empty list")
            outs_LP = outs_LP[0]
    else:
        with  rng_sandbox():
            outs_LP = linear_probe_last_layer(
                args=args,
                model=probe_model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                bs_probe=bs_probe,
                device=str(device_probe)  # for example "cpu" or "cuda:1"
            )

    return {
        "acc_test_retain": outs_LP.get("acc_test_retain"),
        "acc_test_forget": outs_LP.get("acc_test_forget"),
        "best_train_acc": outs_LP.get("best_train_acc"),
        "epochs_trained": outs_LP.get("epochs_trained"),
        "history": outs_LP.get("history"),
    }
    
    return


def linear_probe_CMF_RemoveFC(args, model, train_loader, test_loader,
                             num_classes, bs_probe, device='cpu'):
    """
    Only register hook to layer4.1.bn2 of resnet18，
    Collect features from this layer after pooling for linear probing。
    """
    # 1) Register hook to layer4.1.bn2
    acts = {'layer4.1.bn2': []}
    handle = model.encoder.layer4[1].bn2.register_forward_hook(
        lambda m, inp, out: acts['layer4.1.bn2'].append(
            pool_tensor(out, mode='flatten').detach().cpu()
        )
    )
    hooks = [handle]

    with torch.no_grad():
        feat_tr, lab_tr = collect_features(model, train_loader, device, acts)
        feat_te, lab_te = collect_features(model, test_loader,  device, acts)
    clear_hooks(hooks)

    # prepare constant label tensors once
    labels_tr = torch.tensor(lab_tr, dtype=torch.long)
    labels_te = torch.tensor(lab_te, dtype=torch.long)
    forget_cls = torch.tensor(args.unlearn_class)
    def make_masks(y):
        is_forget = (y.unsqueeze(0)==forget_cls.unsqueeze(1)).any(0)
        return ~is_forget, is_forget

    results = []
    # 2) iterate per layer, convert only this layer to tensors
    for layer in feat_tr.keys():
        # convert numpy to CPU tensor
        Xtr = torch.from_numpy(feat_tr[layer]).float()
        Xte = torch.from_numpy(feat_te[layer]).float()
        retain_tr, forget_tr = make_masks(labels_tr)
        retain_te, forget_te = make_masks(labels_te)

        C = Xtr.size(1)
        #torch.manual_seed(args.seed)
        clf = nn.Linear(C, num_classes).to(device)
        opt = optim.SGD(clf.parameters(), lr=0.1, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        # DataLoader on CPU tensors
        ds = TensorDataset(Xtr, labels_tr)
        probe_loader = DataLoader(ds, batch_size=bs_probe,
                                  shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        
        # early stopping setup
        best_acc = 0.0
        patience = 10
        no_improve = 0
        max_epochs = 150

        history = []  # list of dict: {'epoch', 'test_acc_retain', 'test_acc_forget', 'train_acc_all'}
        
        # train
        for epoch in range(max_epochs):
            clf.train()
            for bx, by in probe_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = loss_fn(clf(bx), by)
                loss.backward()
                opt.step()
            
            # evaluate overall train accuracy
            clf.eval()
            with torch.no_grad():
                pr_tr = clf(Xtr.to(device)).argmax(dim=1).cpu()
                pr_te = clf(Xte.to(device)).argmax(dim=1).cpu()

            acc_train_all = (pr_tr == labels_tr).float().mean().item()

            def masked_acc(pred, y, mask):
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    return None
                return float((pred[idx] == y[idx]).float().mean().item())
            
            acc_te_retain = masked_acc(pr_te, labels_te, retain_te)
            acc_te_forget = masked_acc(pr_te, labels_te, forget_te)

            history.append({
                'epoch': epoch,
                'test_acc_retain': acc_te_retain,
                'test_acc_forget': acc_te_forget,
                'train_acc_all': acc_train_all,
            })

            if acc_train_all > best_acc:
                best_acc = acc_train_all
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        # Final evaluation on both train and test
        clf.eval()
        with torch.no_grad():
            pr_tr = clf(Xtr.float().to(device)).argmax(1).cpu().numpy()
            pr_te = clf(Xte.float().to(device)).argmax(1).cpu().numpy()

        fm_tr = np.isin(lab_tr, args.unlearn_class)
        rm_tr = ~fm_tr
        fm_te = np.isin(lab_te, args.unlearn_class)
        rm_te = ~fm_te



        results.append({
            'layer': layer,
            'acc_train_retain': float((pr_tr[rm_tr] == lab_tr[rm_tr]).mean()) if rm_tr.any() else None,
            'acc_train_forget': float((pr_tr[fm_tr] == lab_tr[fm_tr]).mean()) if fm_tr.any() else None,
            'acc_test_retain': float((pr_te[rm_te] == lab_te[rm_te]).mean()) if rm_te.any() else None,
            'acc_test_forget': float((pr_te[fm_te] == lab_te[fm_te]).mean()) if fm_te.any() else None,
            'best_train_acc': best_acc,
            'epochs_trained':   epoch + 1,
            'history': history
        })
        '''''
        print(f"\n[Linear Probe] Layer: {layer}")
        print("Epoch\tTrainAcc(All)\tTestAcc(Retain)\tTestAcc(Forget)")
        for h in history:
            tr_a = f"{h['train_acc_all']:.4f}"
            te_r = "None" if h['test_acc_retain'] is None else f"{h['test_acc_retain']:.4f}"
            te_f = "None" if h['test_acc_forget'] is None else f"{h['test_acc_forget']:.4f}"
            print(f"{h['epoch']:>3d}\t{tr_a}\t\t{te_r}\t\t{te_f}")
        '''''
    return results

def _expected_len(loader):
    # If DataLoader might have drop_last=True, do not use len(dataset)
    try:
        return len(loader.dataset)
    except Exception:
        return None  # fallbackto“number”to/foraligncheck

def _sanity_counts(name, feat_dict, labs, loader, layer='layer4.1.bn2'):
    n_feat = feat_dict[layer].shape[0]
    n_lab  = len(labs)
    n_exp  = _expected_len(loader)

    print(f"[{name}] expected={n_exp}, feats={n_feat}, labels={n_lab}")
    if n_exp is not None:
        assert n_feat == n_exp == n_lab, f"{name}: Count mismatch, possibly acts not cleared or hook path incorrect"
    else:
        assert n_feat == n_lab, f"{name}: Feature and label count mismatch"


def linear_probe_last_layer(args, model, train_loader, test_loader,
                             num_classes, bs_probe, device='cpu'):
    """
    Only register hook to layer4.1.bn2 of resnet18，
    Collect features from this layer after pooling for linear probing。
    """
    # 1) Register hook to layer4.1.bn2
    acts = {'layer4.1.bn2': []}
    handle = model.layer4[1].bn2.register_forward_hook(
        lambda m, inp, out: acts['layer4.1.bn2'].append(
            pool_tensor(out, mode='avg').detach().cpu()
        )
    )
    hooks = [handle]

    # 2) Collect features using collect_features
    model.eval()
    with torch.no_grad():
        feat_tr, lab_tr = collect_features(model, train_loader, device, acts)
        _sanity_counts("TRAIN", feat_tr, lab_tr, train_loader)

        feat_te, lab_te = collect_features(model, test_loader,  device, acts)
        _sanity_counts("TEST", feat_te, lab_te, test_loader)

    # 3) Unload hooks
    clear_hooks(hooks)

    


    # 4) Keep only features from this layer
    layer = 'layer4.1.bn2'
    Xtr = torch.from_numpy(feat_tr[layer]).float().to(device)  # [N, C]
    Xte = torch.from_numpy(feat_te[layer]).float().to(device)
    labels_tr = torch.tensor(lab_tr, dtype=torch.long).to(device)
    labels_te = torch.tensor(lab_te, dtype=torch.long).to(device)


    #  Intermediate layer output analysis (for checking if changed)
    feat_sample = feat_te[layer]  # numpy, shape [N, C]
    feat_tensor = torch.from_numpy(feat_sample)

    checksum = {
        'mean': feat_tensor.mean().item(),
        'std': feat_tensor.std().item(),
        'sum': feat_tensor.sum().item(),
        'max': feat_tensor.max().item(),
        'min': feat_tensor.min().item()
    }
    print(f"[Feature Checksum for {layer}] mean={checksum['mean']:.6f}, std={checksum['std']:.6f}, sum={checksum['sum']:.6f}")

    # 5) Build and train linear classifier
    clf = nn.Linear(Xtr.size(1), num_classes).to(device)
    opt = optim.SGD(clf.parameters(), lr=args.lr, momentum=args.momentum,
    weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(Xtr, labels_tr)
    loader = DataLoader(ds, batch_size=bs_probe, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))

    best_acc, no_imp, patience, max_epochs = 0.0, 0, 100, 200
    for epoch in range(max_epochs):
        clf.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss_fn(clf(bx), by).backward()
            opt.step()

        # early stopping based on training set accuracy
        clf.eval()
        with torch.no_grad():
            preds = clf(Xtr).argmax(dim=1)
            acc = (preds == labels_tr).float().mean().item()
        if acc > best_acc:
            best_acc, no_imp = acc, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"Early stopping at epoch {epoch+1}, best acc: {best_acc:.4f}")
            break

    # 6) Final evaluation on train/test
    clf.eval()
    with torch.no_grad():
        pr_tr = clf(Xtr).argmax(1).cpu().numpy()
        pr_te = clf(Xte).argmax(1).cpu().numpy()

    fm_tr = np.isin(lab_tr, args.unlearn_class); rm_tr = ~fm_tr
    fm_te = np.isin(lab_te, args.unlearn_class); rm_te = ~fm_te
    
    return {
        'layer': layer,
        'acc_train_retain': float((pr_tr[rm_tr] == lab_tr[rm_tr]).mean()) if rm_tr.any() else None,
        'acc_train_forget': float((pr_tr[fm_tr] == lab_tr[fm_tr]).mean()) if fm_tr.any() else None,
        'acc_test_retain':  float((pr_te[rm_te] == lab_te[rm_te]).mean())  if rm_te.any() else None,
        'acc_test_forget':  float((pr_te[fm_te] == lab_te[fm_te]).mean())  if fm_te.any() else None,
        'best_train_acc':   best_acc,
        'epochs_trained':   epoch + 1
    }



def block_sort_key(name: str):
    """
    Define sorting key for network layers on x-axis：
    conv1 -> layer1 basic blocks -> layer2 -> layer3 -> layer4 -> fc
    """
    # conv1 layer comes first
    if name == "conv1":
        return (0, 0)
    # order of basic blocks in each layer
    for layer_idx in range(1, 5):
        prefix = f"layer{layer_idx}."
        if name.startswith(prefix):
            # Extract basic block index
            parts = name.split('.')
            block_idx = int(parts[1])
            # conv1 and conv2 at same position
            return (layer_idx, block_idx)
    # Fully connected layer at end


    if name == "fc":
        return (5, 0)
    # Other unknown layers at even further end
    return (6, 0)
    # fallback
    return (7, 0)


def plot_LP(args, out):
    layers = sorted(out['layer'], key=block_sort_key)
    acc_ret_train = [out.loc[out['layer']==l, 'acc_train_retain'].values[0] for l in layers]
    acc_for_train = [out.loc[out['layer']==l, 'acc_train_forget'].values[0] for l in layers]
    acc_ret_test  = [out.loc[out['layer']==l, 'acc_test_retain'].values[0]  for l in layers]
    acc_for_test  = [out.loc[out['layer']==l, 'acc_test_forget'].values[0]   for l in layers]

    suffix = ""
    if args.freeze_except_last:
        suffix += "_freeze"
    if getattr(args, "zero_last_layer", False):
        suffix += "_zero"
    method = args.unlearn_method + suffix
    plt_path = f"./plots/{method}/{args.dataset}_{args.arch}/{','.join([str(v) for v in args.unlearn_class])}.png"

    

    if not os.path.exists(plt_path):
        os.makedirs(os.path.dirname(plt_path), exist_ok=True)
    title_base = f"{args.unlearn_method} | {args.dataset} | {args.arch} | forget={','.join([str(v) for v in args.unlearn_class])}"
    plt.figure(figsize=(12,6))
    plt.plot(layers, acc_ret_train, marker='o', label='Train Retain')
    plt.plot(layers, acc_for_train, marker='s', label='Train Forget')
    plt.plot(layers, acc_ret_test,  marker='^', label='Test Retain')
    plt.plot(layers, acc_for_test,  marker='v', label='Test Forget')
    plt.xticks(rotation=90)
    plt.ylabel('Accuracy (%)')
    plt.title(f"{title_base}")
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(plt_path)
    plt.close()
    print(f"Saved plot: {plt_path}")