# =========================
# Linear Probe - Unified Kit
# =========================
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import random, contextlib
from typing import Tuple
import torch
import torch.nn.functional as F
from evaluation.alignment import collect_features_for_head

# ---------- RNG sandbox：ensureoncenotpolluteexternalrandomstate ----------
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

# ---------- [A] more's/oflayerparsedevice：selflogic encoder prefix/leadingdot ----------
def _resolve_module(model, dotted: str, return_name: bool = False):
    """
    parse 'encoder.layer4.1.bn2' or 'layer4.1.bn2' path。
    - tolerateremoveleadingdot（'.layer4.1.bn2' → 'layer4.1.bn2'）
    - if model has no encoder andpath 'encoder.'，tryremoveprefix
    - modelhave encoder andpathdoes not contain 'encoder.'，tryselfaddprefix
    formtimeReturn；return_name=True timeReturnactually/realuseuse's/ofpathcharacterstring。
    """
    def _traverse(obj, path: str):
        cur = obj
        for part in path.split('.'):
            if not part:
                continue
            if part.isdigit():
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
        return cur

    dotted = dotted.lstrip('.')  # tolerate：removeleadingdot
    candidates = [dotted]

    has_enc = hasattr(model, 'encoder')
    if dotted.startswith('encoder.') and not has_enc:
        candidates.append(dotted[len('encoder.'):])
    if (not dotted.startswith('encoder.')) and has_enc:
        candidates.append('encoder.' + dotted)

    last_err = None
    for name in candidates:
        try:
            mod = _traverse(model, name)
            return (mod, name) if return_name else mod
        except Exception as e:
            last_err = e
            continue
    raise AttributeError(f"Cannot resolve layer path among candidates: {candidates}") from last_err


@torch.no_grad()
def _extract_head_input_features(
    model,
    loader,
    device: torch.device,
    *,
    pool_mode: str = "avg",     # 'avg' | 'max' | 'flatten'，onlyin 4D→2D needmatch d_head timegenerate
    normalize: bool = False,    # issamplethis L2 normoneify（LP passcan False；canopen）
) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
    """
    call collect_features_for_head，“classification headoutputinput”feature：
      - X: [N, d_head]（dimensiondegreeselfadaptto head.weight.shape[1]）
      - y: [N]
      - used: getcome（'head::<input>::...' orfallbackcome）
      - d_used: actually/realfeaturedimensiondegree（applyetcin d_head）
    """
    feats, labels, used, d_used = collect_features_for_head(
        model=model,
        loader=loader,
        device=device,
        pool_mode=pool_mode,
        normalize=normalize,
    )
    return feats, labels, used, d_used

def _extract_layer_features_vit(model, dataloader, device):
    model.eval()
    feats_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            #images = images.to(device)
            #labels_list.append(labels.cpu())

            # ---- open ViT 's/of forward，in heads before ----
            # 1) patch + conv_proj
            #x = model._process_input(images)   # (B, seq_len, hidden_dim)

            # 2)  class token
            #n = x.shape[0]
            #batch_class_token = model.class_token.expand(n, -1, -1)  # (B, 1, hidden_dim)
            #x = torch.cat([batch_class_token, x], dim=1)             # (B, seq_len+1, hidden_dim)

            # 3) encoder
            #x = model.encoder(x)    # (B, seq_len+1, hidden_dim)

            #if hasattr(model, "norm") and model.norm is not None:
            #    x = model.norm(x)

            # 4) get CLS token do feature
            #cls_feat = x[:, 0]      # (B, hidden_dim)

            #feats_list.append(cls_feat.cpu())
            images = images.to(device, non_blocking=True)
            feat = model.get_head_input(images)      # ★unified/statisticoneinput
            feats_list.append(feat.detach().cpu())
            labels_list.append(labels.cpu())


    features = torch.cat(feats_list, dim=0)   # (N, hidden_dim)
    labels   = torch.cat(labels_list, dim=0)  # (N,)
    return features, labels


# ---------- [B] unified/statisticone：featureextractget（support 4D/2D output） ----------
@torch.no_grad()
def _extract_layer_features(model, loader, layer_name='encoder.layer4.1.bn2',
                            device='cpu', pool='avg'):
    """
    infixedlayer forward hook，extractgetlayeroutputanddopooling/flatten：
      - pool='avg'     → to/for 4D outputdo GAP to [B,C]
      - pool='flatten' → to/for 4D outputflattento [B, C*H*W]
      - to/for 2D output（propertylayer）directlyuseuse
    Return：X [N,D] (float32, CPU), y [N] (long, CPU)
    """
    target = _resolve_module(model, layer_name)
    feats, labels = [], []

    def _hook(m, inp, out):
        if out.dim() == 4:
            if pool == 'avg':
                f = F.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1)  # [B,C]
            elif pool == 'flatten':
                f = out.flatten(1)                                              # [B, C*H*W]
            else:
                raise ValueError(f"Unknown pool mode: {pool}")
        elif out.dim() == 2:
            f = out  # [B,D]
        else:
            f = out.flatten(1)  # 
        feats.append(f.detach().cpu())

    handle = target.register_forward_hook(_hook)

    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        labels.append(y.detach().cpu())

    handle.remove()

    X = torch.cat(feats, dim=0).float()   # [N,D] CPU float32
    y = torch.cat(labels, dim=0).long()   # [N]   CPU long
    # optional：numbercall
    # print(f"[Extract] layer={layer_name}, feats={X.size(0)}, labels={y.size(0)}")
    return X, y

# ---------- [C] unified/statisticone：propertyclassificationdevicetraining（val ；notlook test） ----------
def _train_linear_probe(Xtr, ytr, num_classes, device='cpu',
                        seed=42, lr=0.1, momentum=0.9, weight_decay=0.0,
                        batch_size=256, patience=50, max_epochs=200, val_ratio=0.1, use_cosine=False):
    """
    inpropertyfeaturetrainingpropertyclassificationdevice：
      - model：nn.Linear(D, num_classes)
      - separatetraining/validation（val_ratio），validationpreparesure
      - Return (clf, best_val_acc, best_epoch, history)
        history onlyrecord train_all_acc and val_acc（ test ）
    """
    D = Xtr.size(1)
    clf = nn.Linear(D, num_classes).to(device)

    n_total = Xtr.size(0)
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 and n_total >= 10 else 0
    if n_val > 0:
        gen = torch.Generator().manual_seed(seed)
        ds_all = TensorDataset(Xtr, ytr)
        ds_tr, ds_val = random_split(ds_all, [n_total - n_val, n_val], generator=gen)
        dl_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  generator=gen)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    else:
        dl_tr  = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        dl_val = None

    opt = optim.SGD(clf.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sch = None
    if use_cosine:
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch = -1.0, None, 0
    no_imp, history = 0, []

    for epoch in range(1, max_epochs + 1):
        clf.train()
        for bx, by in dl_tr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = loss_fn(clf(bx), by)
            loss.backward()
            opt.step()

        # trainingpreparesure（record）
        clf.eval()
        with torch.no_grad():
            pred_tr = clf(Xtr.to(device)).argmax(1).cpu()
            if ytr.device != pred_tr.device:
                ytr_dev = ytr.to(pred_tr.device)
            else:
                ytr_dev = ytr

            train_acc = float((pred_tr == ytr_dev).float().mean().item())

        # validationpreparesureis/be
        if dl_val is not None:
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in dl_val:
                    pr = clf(bx.to(device)).argmax(1).cpu()
                    if by.device != pr.device:
                        by = by.to(pr.device)
                    corr += int((pr == by).sum().item())
                    tot  += int(by.numel())
            val_acc = corr / max(1, tot)
        else:
            val_acc = train_acc  # validationtimeifyis/betrainingpreparesure

        history.append({'epoch': epoch, 'train_acc_all': train_acc, 'val_acc': val_acc})

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
            best_epoch = epoch
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            print(f"[EarlyStop] epoch={epoch}, best_val={best_val:.4f}")
            break
        if sch is not None:
            sch.step()


    if best_state is not None:
        clf.load_state_dict(best_state)
    return clf, float(best_val if best_val >= 0 else 0.0), int(best_epoch), history

# ---------- [D] unified/statisticone：evaluate retain/forget ----------
@torch.no_grad()
def _eval_masks(clf, X, y, device, forget_classes):
    pred = clf(X.to(device)).argmax(1).cpu().numpy()
    true = y.cpu().numpy()
    if forget_classes is None:
        forget_classes = []
    fm = np.isin(true, np.array(forget_classes, dtype=true.dtype)) if len(forget_classes) > 0 else np.zeros_like(true, dtype=bool)
    rm = ~fm

    def acc(mask):
        return None if mask.size == 0 or not mask.any() else float((pred[mask] == true[mask]).mean())

    return acc(rm), acc(fm)  # (retain_acc, forget_acc)

# ---- new：and CMF consistent's/offeatureextractget ----
@torch.no_grad()
def _extract_cmf_geometry_features(model, loader, device):
    model.eval()
    X_buf, y_buf = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model.extract_features(x)
        z = model._preprocess_feats_for_cmf(f)   # ★ key：ẑ = normalize(f) ->  μ -> normalize
        X_buf.append(z.detach().cpu())
        y_buf.append(y.detach().cpu())
    X = torch.cat(X_buf, dim=0).float()
    y = torch.cat(y_buf, dim=0).long()
    return X, y


# ---------- unified/statisticoneinput：extractget + training + evaluate ----------
def unified_linear_probe(args, model, train_loader, test_loader,
                         num_classes, bs_probe, *,
                         layer_name='encoder.layer4',
                         pool='avg', device='cpu', normalize_feats_for_lp=False):
    """
    unified/statisticoneprepare's/ofproperty：
      - fixed layer_name andpoolingstyle（'avg'/'flatten'）
      - useunified/statisticone's/offeatureextractget + training + evaluate
    Returnunified/statisticonecharacter：
      layer/pool/acc_train_retain/acc_train_forget/acc_test_retain/acc_test_forget/
      best_train_acc/best_val_acc/epochs_trained/history
    """
    # parseoncelayer name，recordactually/realuseuse's/ofnamecharacter
    # ---- ViT：use CLS feature，not module path ----
    if ("vit" in args.arch.lower()) or (layer_name in ["vit_cls", "cls_token"]):
        resolved = "vit_cls"
    else:
        _, resolved = _resolve_module(model, layer_name, return_name=True)

    print(f"[LP] arch={args.arch} dataset={args.dataset} layer_name request={layer_name} resolved={resolved}")
    #mod = dict(model.named_modules()).get(resolved, None)
    #print("[LP] resolved module =", type(mod), mod)

    # extractgetfeature（train/test notpollute）
    if "CMF" in args.unlearn_method:
        #print(f"[LP] Using CMF geometry feature extraction for layer={resolved}, pool={pool}, device={device}")
        Xtr, ytr = _extract_cmf_geometry_features(model, train_loader, device)
        Xte, yte = _extract_cmf_geometry_features(model, test_loader,  device)
        d_tr, d_te = Xtr.size(1), Xte.size(1)
        assert d_tr == d_te, f"Train/Test feature dims mismatch: {d_tr} vs {d_te}"
    else:
        if "vit" in args.arch.lower():
            Xtr, ytr = _extract_layer_features_vit(model, train_loader, device)
            Xte, yte  = _extract_layer_features_vit(model, test_loader,  device)
        else:
            Xtr, ytr = _extract_layer_features(model, train_loader, resolved, device, pool)
            Xte, yte = _extract_layer_features(model, test_loader,  resolved, device, pool)
        

    #Xtr, ytr, used_tr, d_tr = _extract_head_input_features(model, train_loader, device, pool_mode=pool, normalize=normalize_feats_for_lp)
    #Xte, yte, used_te, d_te = _extract_head_input_features(model, test_loader, device, pool_mode=pool, normalize=normalize_feats_for_lp)

    #assert d_tr == d_te, f"Train/Test feature dims mismatch: {d_tr} vs {d_te}"

    print("[LP] Xtr shape:", Xtr.shape, "dtype:", Xtr.dtype, "device:", Xtr.device)
    print("[LP] stats: mean/std/min/max =",
        float(Xtr.mean()), float(Xtr.std()), float(Xtr.min()), float(Xtr.max()))
    print("[LP] nan?", torch.isnan(Xtr).any().item(), "inf?", torch.isinf(Xtr).any().item())

    # againlooklookclasssamplethisnumberis（200classat leastnotisonlyclass）
    unique, counts = torch.unique(ytr.cpu(), return_counts=True)
    print("[LP] num_classes_in_probe_train:", unique.numel(), "min_count:", int(counts.min()), "max_count:", int(counts.max()))

    import torch.nn.functional as F

    is_tiny = bool(getattr(args, "tinyimagenet", False)) or (getattr(args, "dataset", "").lower() == "tinyimagenet")
    is_cifar100 = getattr(args, "dataset", "").lower() == "cifar100"
    is_cifar10 = getattr(args, "dataset", "").lower() == "cifar10"

    # ---- Tiny use：recommendedto/forfeaturedo L2 normalize（not CIFAR）----
    #if is_tiny:
    #    Xtr = F.normalize(Xtr, dim=1)
    #    Xte = F.normalize(Xte, dim=1)

    # ---- Tiny use（not CIFAR ）----
    #lp_lr         = 0.1  if is_tiny or is_cifar100 else getattr(args, "lr", 0.01)
    #lp_wd         = 1e-4  if is_tiny or is_cifar100 else getattr(args, "weight_decay", 0.0)
    lp_lr         = 0.01 
    #lp_wd         = 1e-4  if is_tiny or is_cifar100 else getattr(args, "weight_decay", 0.0)
    lp_patience   = 40    
    lp_max_epochs =150   if (is_tiny or is_cifar100) else 20 if is_cifar10 else getattr(args, "max_epochs", 200)
    lp_val_ratio  = 0.1  


    clf, best_val_acc, best_epoch, history = _train_linear_probe(
        Xtr, ytr, num_classes,
        device=device,
        seed=getattr(args, "seed", 42),
        lr=lp_lr,
        momentum=getattr(args, "momentum", 0.9),
        #weight_decay=lp_wd,
        weight_decay=0.0,  # 
        batch_size=bs_probe,
        patience=lp_patience,
        max_epochs=lp_max_epochs,
        val_ratio=lp_val_ratio,
        use_cosine=is_tiny
    )
    

    # evaluate retain/forget（train & test）
    forget_classes = getattr(args, 'unlearn_class', [])
    acc_tr_retain, acc_tr_forget = _eval_masks(clf, Xtr, ytr, device, forget_classes)
    acc_te_retain, acc_te_forget = _eval_masks(clf, Xte, yte, device, forget_classes)
    print(f"[LP] Train Acc - Retain: {acc_tr_retain}, Forget: {acc_tr_forget}")
    print(f"[LP] Test  Acc - Retain: {acc_te_retain}, Forget: {acc_te_forget}")

    return {
        'layer': resolved,
        'pool': pool,
        'acc_train_retain': acc_tr_retain,
        'acc_train_forget': acc_tr_forget,
        'acc_test_retain':  acc_te_retain,
        'acc_test_forget':  acc_te_forget,
        'best_train_acc':   float(history[best_epoch-1]['train_acc_all']) if history and best_epoch >= 1 else None,
        'best_val_acc':     float(best_val_acc),
        'epochs_trained':   int(best_epoch),
        'history':          history,  # only train/val（does not contain test），
    }

def _is_resnet_arch(args):
    return "resnet" in getattr(args, "arch", "").lower()

# ---------- accordingmodelconstructselffixedlayer name ----------
def _default_layer_name_for_model(args,model, base='layer4'):
    if _is_resnet_arch(args):
        base = "layer4"
    return f'encoder.{base}' if hasattr(model, 'encoder') else base

# ---------- layer：firstmodelagaindoproperty ----------
def run_linear_probe_on_fresh_clone(
    args, get_model_fn, device_probe, src_model,
    train_loader, test_loader, num_classes, bs_probe,
    *, layer_name='auto', pool=None
):
    """
    firstone“only”model，againbyunified/statisticonepreparedoproperty：
      1) get_model_fn createnewmodelandweight
      2) number、eval style
      3) chooselayer name：layer_name='auto' → selfaccordingmodelconstructfixedisadd 'encoder.'
      4) choosepooling：stylefixedtime，CMF methoduse flatten，otheruse avg
      5) calluse unified_linear_probe，Returnunified/statisticoneconstruct's/of
    """
    # newusemodel，and state_dict
    probe_model = get_model_fn(args, device_probe)
    print("[LP] New probe model created.")
    probe_model.load_state_dict(src_model.state_dict(), strict=True)
    print("[LP] State dict loaded into probe model.")

    #  + eval
    probe_model.eval()
    for p in probe_model.parameters():
        p.requires_grad_(False)

    # layer name：auto → accordingmodelconstructchoose 'encoder.layer4.1.bn2' or 'layer4.1.bn2'
    if layer_name == 'auto' or not layer_name:
        layer_name = _default_layer_name_for_model(args, probe_model)

    # pooling：stylefixedtimebymethodname
    if pool is None:
        pool = 'avg' 

    print(f"[LP] layer={layer_name}, pool={pool}, device={device_probe}")

    with rng_sandbox():
        outs = unified_linear_probe(
            args=args,
            model=probe_model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            bs_probe=bs_probe,
            layer_name=layer_name,
            pool=pool,
            device=str(device_probe)
        )

    return {
        "layer":             outs["layer"],     # actually/realparseafter's/oflayer name
        "pool":              outs["pool"],
        "acc_train_retain":  outs["acc_train_retain"],
        "acc_train_forget":  outs["acc_train_forget"],
        "acc_test_retain":   outs["acc_test_retain"],
        "acc_test_forget":   outs["acc_test_forget"],
        "best_train_acc":    outs["best_train_acc"],
        "best_val_acc":      outs["best_val_acc"],
        "epochs_trained":    outs["epochs_trained"],
        "history":           outs["history"],
    }

# ---------- 's/of（optional） ----------
def linear_probe_last_layer(args, model, train_loader, test_loader,
                            num_classes, bs_probe, device='cpu'):
    """
    name：useuse GAP（avg）to/for layer4.1.bn2 do LP。
    """

    print("Warning: linear_probe_last_layer is deprecated. Use unified_linear_probe instead.")
    layer_name = _default_layer_name_for_model(args, model)
    
    return unified_linear_probe(
        args, model, train_loader, test_loader,
        num_classes, bs_probe,
        layer_name=layer_name,
        pool='avg',
        device=device
    )

def linear_probe_CMF_RemoveFC(args, model, train_loader, test_loader,
                              num_classes, bs_probe, device='cpu'):
    """
    name：useuse flatten to/for layer4.1.bn2 do LP（ CMF_RemoveFC fixed）。
    """
    layer_name = _default_layer_name_for_model(args, model)
    return unified_linear_probe(
        args, model, train_loader, test_loader,
        num_classes, bs_probe,
        layer_name=layer_name,
        pool='flatten',
        device=device
    )
