import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from unlearn.tools import apply_prep
import evaluation


def get_salun_mask(args, model, device, forget_loader):
    
    if args.unlearn_method != "salun":
        return None 
    mask = {}
    for name, param in model.named_parameters():
        mask[name] = torch.zeros_like(param, device=device)

    was_training = model.training
    model.eval()
    for data, target in forget_loader:
        data, target = data.to(device), target.to(device)

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(data)                          # passis log-probs
            loss = F.nll_loss(output, target, reduction='sum')  # andyouoriginalconsistent
        loss.backward()

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    mask[name] += p.grad.detach().abs()

    all_elements = -torch.cat([t.flatten() for t in mask.values()])
    threshold_index = int(len(all_elements) * args.salun_threshold)
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    hard_dict, start = {}, 0
    for key, tensor in mask.items():
        n = tensor.numel()
        local_ranks = ranks[start:start+n].reshape(tensor.shape)
        start += n
        h = torch.zeros_like(tensor)
        h[local_ranks < threshold_index] = 1
        hard_dict[key] = h
    return hard_dict

@apply_prep
def salun_unlearn(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs, **kwargs
):
    """
    SALUN unlearning with unified evaluation:
      - build SALUN mask on forget set
      - mix (subset of) forget/retain, random-relabel forget classes
      - SGD with masked gradients
      - epoch-0 baseline test (+ optional LP)
      - test (+ optional LP) after each epoch
    """
    from utils import test, get_model
    import torch.optim as optim

    # Optimizer (re-create to ensure settings)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )

    num_classes = args.num_classes
    forget_classes = set(args.unlearn_class)

    # ---------- Build mixed training loader (fixed subsets) ----------
    forget_dataset = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(forget_dataset)
    forget_dataset, _ = torch.utils.data.random_split(
        forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples]
    )

    retain_dataset = retain_loader.dataset
    assert args.num_retain_samples <= len(retain_dataset)
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples]
    )

    train_dataset = ConcatDataset([forget_dataset, retain_dataset])
    salun_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(forget_loader, "num_workers", 0),
        pin_memory=True
    )

    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    # ---------- Logs ----------
    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list, LP_history_list = [], [], []
    epoch_list = [0] + list(range(1, epochs + 1))
    lp_every = getattr(args, "lp_every", 1)  # 0 to disable LP

    # ---------- Epoch-0 baseline ----------
    model.eval()

    print("[Before Unlearning] Evaluating (epoch 0 baseline)")
    retain_acc, forget_acc, _ = test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (Epoch 0)"
    )
    retain_acc_list.append(retain_acc)
    forget_acc_list.append(forget_acc)

    if lp_every != 0:
        print("[LP] Running linear probe at epoch 0...")
        outs_LP0 = evaluation.run_linear_probe_on_fresh_clone(
            args=args,
            get_model_fn=get_model,
            device_probe=device,
            src_model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=args.num_classes,
            bs_probe=getattr(args, "prob_batch_size",
                             getattr(args, "probe_batch_size", 256)),
        )
        LP_retain_acc_list.append(outs_LP0["acc_test_retain"])
        LP_forget_acc_list.append(outs_LP0["acc_test_forget"])
        LP_history_list.append(outs_LP0.get("history"))
        print(f"[LP@epoch0] retain={outs_LP0['acc_test_retain']}, "
              f"forget={outs_LP0['acc_test_forget']}")

    # ---------- Build SALUN mask ----------
    mask = get_salun_mask(args, model, device, forget_loader)
    
    # ---------- Training epochs ----------
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}]")
        model.train()

        for data, target in salun_train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # Random relabel only for forget classes
            forget_mask = torch.zeros_like(target, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (target == cls)

            if forget_mask.any():
                n_forget = int(forget_mask.sum().item())
                rand = choices[torch.randint(0, len(valid), (n_forget,), device=device)]
                target[forget_mask] = rand

            optimizer.zero_grad(set_to_none=True)
            output = model(data)                         # logits
            loss = F.nll_loss(output, target)       # CE fits logits
            loss.backward()

            # Apply SALUN mask to grads (element-wise)
            if mask is not None:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is not None and name in mask:
                            p.grad.mul_(mask[name])

            optimizer.step()

            if getattr(args, "dry_run", False):
                break

        # ----- Eval after this epoch -----
        model.eval()

        retain_acc, forget_acc, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

        # Optional LP
        if lp_every > 0 and (epoch % lp_every == 0):
            print(f"[Epoch {epoch}] Evaluating Linear Probe...")
            outs_LP = evaluation.run_linear_probe_on_fresh_clone(
                args=args,
                get_model_fn=get_model,
                device_probe=device,
                src_model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=getattr(args, "prob_batch_size",
                                 getattr(args, "probe_batch_size", 256)),
            )
            LP_retain_acc_list.append(outs_LP["acc_test_retain"])
            LP_forget_acc_list.append(outs_LP["acc_test_forget"])
            LP_history_list.append(outs_LP.get("history"))
            print(f"[LP@epoch{epoch}] retain={outs_LP['acc_test_retain']}, "
                  f"forget={outs_LP['acc_test_forget']}")

        model.train()

    # ---------- Save history ----------
    model.history_log = {
        "retain_acc":    retain_acc_list,
        "forget_acc":    forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "LP_history":    LP_history_list,
        "epoch":         epoch_list,
    }
    return model



import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import copy
import evaluation
from unlearn.tools import apply_prep

def build_salun_mask(args, model, device, forget_loader):
    """
    Build salun's gradient-importance mask on forget set.
    - Avoid BN/Dropout pollution: collect in model.eval().
    - Compute grads under `torch.enable_grad()`.
    - Use CrossEntropy on logits (NOT NLL).
    """
    if args.unlearn_method not in ("salun", "salun_CMF_RemoveFC"):
        return None

    # Accumulate per-parameter gradient magnitudes
    mask = {name: torch.zeros_like(p, device=device)
            for name, p in model.named_parameters()}

    was_training = model.training
    model.eval()  # keep BN running stats intact

    for data, target in forget_loader:
        data, target = data.to(device), target.to(device)

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(data)                               # logits
            loss = F.cross_entropy(logits, target, reduction='sum')
        loss.backward()

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    mask[name] += p.grad.detach().abs()        # accumulate |grad|

    model.train(was_training)

    # Global top-p hard mask (1 = keep grad, 0 = drop)
    flat = -torch.cat([t.flatten() for t in mask.values()])    # negative -> descending by abs
    kth = int(len(flat) * args.salun_threshold)                 # e.g., 0.5 keeps top 50%
    idx_sorted = torch.argsort(flat)
    ranks = torch.argsort(idx_sorted)

    hard = {}
    start = 0
    for name, t in mask.items():
        n = t.numel()
        local_ranks = ranks[start:start+n].reshape(t.shape)
        start += n

        h = torch.zeros_like(t, dtype=t.dtype, device=t.device)
        h[local_ranks < kth] = 1.0
        hard[name] = h

    return hard


@apply_prep
def salun_CMF_unlearn(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    **kwargs
):
    """
    SALUN + CMF:
      - Random relabeling for forget classes (only those).
      - Apply salun hard mask to gradients.
      - After each epoch: recompute CMF on train_loader (true labels) in eval mode.
      - Evaluate forward accuracy; optional LP.
    """
    from utils import test, get_model

    # Optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    clip = getattr(args, "grad_norm_clip", None)

    # ---- Epoch 0: CMF align + baseline eval (+ LP) ----
    print("[CMF] Recomputing class means (epoch 0)...")
    model.eval()
    model.recompute_cmf(train_loader, device=device)


    
    print("[Eval] Epoch 0 baseline...")
    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list, LP_history_list = [], [], []
    epoch_list = list(range(0, epochs + 1))

    retain_acc, forget_acc, _ = test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (Epoch 0)"
    )
    retain_acc_list.append(retain_acc)
    forget_acc_list.append(forget_acc)

    lp_every = getattr(args, "lp_every", 1)
    if lp_every != 0:
        print("[LP] Running linear probe at epoch 0...")
        outs_LP0 = evaluation.run_linear_probe_on_fresh_clone(
            args=args, get_model_fn=get_model, device_probe=device,
            src_model=model,
            train_loader=train_loader, test_loader=test_loader,
            num_classes=args.num_classes,
            bs_probe=getattr(args, "prob_batch_size",
                             getattr(args, "probe_batch_size", 256)),
        )
        LP_retain_acc_list.append(outs_LP0["acc_test_retain"])
        LP_forget_acc_list.append(outs_LP0["acc_test_forget"])
        LP_history_list.append(outs_LP0.get("history"))
        print(f"[LP@epoch0] retain={outs_LP0['acc_test_retain']}, "
              f"forget={outs_LP0['acc_test_forget']}")

    # ---- Build salun mask on forget set ----
    print("[SALUN] Building gradient-importance mask on forget set...")
    hard_mask = build_salun_mask(args, model, device, forget_loader)

    # ---- Fixed mixed loader (subset once) ----
    num_classes = args.num_classes
    forget_classes = set(args.unlearn_class)

    full_forget = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(full_forget), \
        "num_forget_samples larger than forget_dataset size"
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget,
        [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f
    )

    full_retain = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(full_retain), \
        "num_retain_samples larger than retain_dataset size"
    gen_r = torch.Generator().manual_seed(args.seed + 1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain,
        [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r
    )

    mixed_loader = DataLoader(
        ConcatDataset([forget_sub, retain_sub]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(forget_loader, "num_workers", 0),
        pin_memory=True
    )

    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    # ---- Train epochs ----
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}]")
        model.train()

        for inputs, labels_true in mixed_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_true = labels_true.to(device, non_blocking=True)

            # random relabel only the forget classes
            labels = labels_true.clone()
            forget_mask = torch.zeros_like(labels_true, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (labels_true == cls)
            if forget_mask.any():
                n_forget = int(forget_mask.sum().item())
                rand_targets = choices[torch.randint(
                    low=0, high=len(valid), size=(n_forget,), device=device
                )]
                labels[forget_mask] = rand_targets

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)                      # logits
            loss = F.cross_entropy(logits, labels)      # CE on fake labels
            loss.backward()

            # apply salun hard mask
            if hard_mask is not None:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is not None and name in hard_mask:
                            p.grad.mul_(hard_mask[name])

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            if getattr(args, "dry_run", False):
                break

        # CMF realign (eval-style) on train_loader (true labels)
        model.eval()
        print("[CMF] Recomputing class means after epoch...")
        model.recompute_cmf(train_loader, device=device)

        # forward eval
        retain_acc, forget_acc, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

        # optional LP
        if lp_every > 0 and (epoch % lp_every == 0):
            print(f"[LP] Evaluating linear probe at epoch {epoch}...")
            outs_LP = evaluation.run_linear_probe_on_fresh_clone(
                args=args, get_model_fn=get_model, device_probe=device,
                src_model=model,
                train_loader=train_loader, test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=getattr(args, "prob_batch_size",
                                 getattr(args, "probe_batch_size", 256)),
            )
            LP_retain_acc_list.append(outs_LP["acc_test_retain"])
            LP_forget_acc_list.append(outs_LP["acc_test_forget"])
            LP_history_list.append(outs_LP.get("history"))
            print(f"[LP@epoch{epoch}] retain={outs_LP['acc_test_retain']}, "
                  f"forget={outs_LP['acc_test_forget']}")
        scheduler.step()
        print(f"[Scheduler] LR after epoch {epoch}: {scheduler.get_last_lr()[0]:.6f}")
        model.train()

    # log
    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "epoch": epoch_list,
    }
    return model
