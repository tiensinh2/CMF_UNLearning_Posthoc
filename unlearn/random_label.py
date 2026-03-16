import torch
import numpy as np
import torch.nn.functional as F
import copy
from unlearn.tools import apply_prep
import evaluation


from unlearn.tools import apply_prep
import copy
import torch
import torch.nn.functional as F

import copy
import torch
import torch.nn.functional as F
from unlearn.tools import apply_prep

@apply_prep
def random_label_CMF_unlearn_iter_eval(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    test_forget_loader, **kwargs
):
    """
    CMF + random-label unlearning WITH iteration-based evaluation (value-only logs),
    keeping EXACTLY the same logging style/keys as random_label_unlearn_iter_eval.

    Training:
      - For samples in forget classes: relabel to random non-forget classes (fake labels).
      - Compute logits using fixed CMF weights (detached) and CE loss on fake labels.
      - Update encoder (and whatever params are trainable).

    Epoch-end:
      - Recompute CMF using train_loader with TRUE labels: model.recompute_cmf(...)
      - Evaluate forward / optional LP / optional NCC (epoch-based)
      - Record epoch_end_iter and epoch_end_* value-only.
    """
    from utils import test, get_model
    import evaluation
    from evaluation.nc_cmf import ncc_mismatch

    # -------------------------
    # Optimizer (same style as your iter-eval baseline)
    # -------------------------
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    clip = getattr(args, "grad_norm_clip", None)

    num_classes = args.num_classes
    forget_classes = set(args.unlearn_class)

    # -------------------------
    # Build subsets once (same as your iter-eval baseline)
    # -------------------------
    full_forget = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(full_forget), "num_forget_samples larger than forget_dataset size"
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f
    )

    full_retain = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(full_retain), "num_retain_samples larger than retain_dataset size"
    gen_r = torch.Generator().manual_seed(args.seed + 1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r
    )

    mixed_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([forget_sub, retain_sub]),
        batch_size=args.batch_size,
        shuffle=True
    )

    # valid target classes: non-forget
    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    # -------------------------
    # Eval frequency (iteration-based)  [SAME AS YOUR BASELINE]
    # -------------------------
    eval_every_iter = getattr(args, "eval_every_iter", 0)   # forward test
    lp_every_iter   = getattr(args, "lp_every_iter", 0)     # LP
    ncc_every_iter  = getattr(args, "ncc_every_iter", 0)    # NCC

    # Keep your epoch-based LP/NCC knobs unchanged
    lp_every_epoch  = getattr(args, "lp_every", 1)   # 0 disables LP entirely (epoch-based)
    ncc_every_epoch = getattr(args, "ncc_every", 1)  # 0 disables NCC entirely (epoch-based)

    # -------------------------
    # Logs (SAME AS YOUR BASELINE)
    # -------------------------
    iter_list = []
    iter_retain_acc_list, iter_forget_acc_list = [], []
    iter_LP_retain_acc_list, iter_LP_forget_acc_list = [], []
    iter_NCC_retain_acc_list, iter_NCC_forget_acc_list = [], []
    iter_NCC_retain_mis_list, iter_NCC_forget_mis_list = [], []

    epoch_list = list(range(0, epochs + 1))
    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    NCC_retain_acc_list, NCC_forget_acc_list = [], []
    NCC_retain_mis_list, NCC_forget_mis_list = [], []

    epoch_end_iter_list = []  # epoch 0 baseline at iter 0
    epoch_end_forget_acc_list, epoch_end_retain_acc_list = [], []
    epoch_end_LP_retain_list, epoch_end_LP_forget_list = [], []
    epoch_end_NCC_retain_list, epoch_end_NCC_forget_list = [], []
    epoch_end_NCC_retain_mis_list, epoch_end_NCC_forget_mis_list = [], []

    # -------------------------
    # Helper evals (SAME CALLS AS YOUR BASELINE)
    # -------------------------
    def _run_forward_test(set_name: str):
        r, f, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=set_name
        )
        return r, f

    def _run_lp():
        outs_lp = evaluation.run_linear_probe_on_fresh_clone(
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
        return outs_lp["acc_test_retain"], outs_lp["acc_test_forget"]

    def _run_ncc():
        pool_mode = getattr(args, "nc_pool_mode", "avg")
        outs_retain = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_loader,
            device=device,
            pool_mode=pool_mode,
        )
        outs_forget = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_forget_loader,
            device=device,
            pool_mode=pool_mode,
        )
        return (outs_retain["ncc_acc"], outs_retain["ncc_mismatch"],
                outs_forget["ncc_acc"], outs_forget["ncc_mismatch"])

    def _maybe_iter_eval(global_iter: int):
        """
        Iteration-based evaluation hooks. (SAME as your baseline)
        - forward test if global_iter % eval_every_iter == 0
        - LP if global_iter % lp_every_iter == 0
        - NCC if global_iter % ncc_every_iter == 0
        """
        
        need = ((eval_every_iter and global_iter % eval_every_iter == 0) or
                (lp_every_iter   and global_iter % lp_every_iter   == 0) or
                (ncc_every_iter  and global_iter % ncc_every_iter  == 0))
        if not need:
            return

        model.eval()

        if eval_every_iter and (global_iter % eval_every_iter == 0):
            r, f = _run_forward_test(set_name=f"Test Set (Iter {global_iter})")
            iter_list.append(global_iter)
            iter_retain_acc_list.append(r)
            iter_forget_acc_list.append(f)
            print(f"[Iter {global_iter}] forward: retain={r:.4f}, forget={f:.4f}")

        if lp_every_iter and (global_iter % lp_every_iter == 0):
            print(f"[Iter {global_iter}] Evaluating Linear Probe (iter)...")
            lp_r, lp_f = _run_lp()
            iter_LP_retain_acc_list.append(lp_r)
            iter_LP_forget_acc_list.append(lp_f)
            print(f"[LP@iter{global_iter}] retain={lp_r:.4f}, forget={lp_f:.4f}")

        if ncc_every_iter and (global_iter % ncc_every_iter == 0):
            print(f"[Iter {global_iter}] Evaluating NCC (iter)...")
            ncc_r, mis_r, ncc_f, mis_f = _run_ncc()
            iter_NCC_retain_acc_list.append(ncc_r)
            iter_NCC_forget_acc_list.append(ncc_f)
            iter_NCC_retain_mis_list.append(mis_r)
            iter_NCC_forget_mis_list.append(mis_f)
            print(f"[NCC@iter{global_iter}] retain={ncc_r:.4f} (mis={mis_r:.4f}) | "
                  f"forget={ncc_f:.4f} (mis={mis_f:.4f})")

        model.train()

    # -------------------------
    # Baseline @ epoch 0 (iter=0)
    # -------------------------
    model.eval()
    # CMF needs geometry aligned first (important for "epoch0" to be meaningful)
    if hasattr(model, "recompute_cmf"):
        model.recompute_cmf(train_loader, device=device)

    print("[Before Unlearning] Evaluating model (epoch 0 baseline)")
    retain_acc0, forget_acc0 = _run_forward_test(set_name="Test Set (Epoch 0)")

    epoch_end_iter_list.append(0)
    epoch_end_retain_acc_list.append(retain_acc0)
    epoch_end_forget_acc_list.append(forget_acc0)

    # ()  epoch_list and retain_acc_list to/foralign：put epoch0 into
    # ifyousupport retain_acc_list does not contain epoch0，canline/execute
    retain_acc_list.append(retain_acc0)
    forget_acc_list.append(forget_acc0)

    if lp_every_epoch != 0:
        print("[LP] Running linear probe at epoch 0...")
        lp_r0, lp_f0 = _run_lp()
        LP_retain_acc_list.append(lp_r0)
        LP_forget_acc_list.append(lp_f0)
        epoch_end_LP_retain_list.append(lp_r0)   # value-only (same style as your baseline)
        epoch_end_LP_forget_list.append(lp_f0)
        print(f"[LP@epoch0] retain={lp_r0:.4f}, forget={lp_f0:.4f}")

    if ncc_every_epoch != 0:
        print("[NCC] Running NCC evaluation at epoch 0...")
        ncc_r0, mis_r0, ncc_f0, mis_f0 = _run_ncc()
        NCC_retain_acc_list.append(ncc_r0)
        NCC_forget_acc_list.append(ncc_f0)
        NCC_retain_mis_list.append(mis_r0)
        NCC_forget_mis_list.append(mis_f0)
        epoch_end_NCC_retain_list.append(ncc_r0)      # value-only
        epoch_end_NCC_forget_list.append(ncc_f0)
        epoch_end_NCC_retain_mis_list.append(mis_r0)
        epoch_end_NCC_forget_mis_list.append(mis_f0)
        print(f"[NCC@epoch0] retain={ncc_r0:.4f} (mis={mis_r0:.4f}) | "
              f"forget={ncc_f0:.4f} (mis={mis_f0:.4f})")

    model.train()

    # -------------------------
    # Training epochs + iteration eval (CMF training logic)
    # -------------------------
    global_iter = 0
    for epoch in range(1, epochs + 1):
        print(f"[Epoch {epoch}]")
        model.train()

        for inputs, labels_true in mixed_loader:
            global_iter += 1
            inputs = inputs.to(device)
            labels_true = labels_true.to(device)

            # ---- fake labels for forget classes (same masking pattern as your baseline)
            labels_train = labels_true.clone()
            forget_mask = torch.zeros_like(labels_true, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (labels_true == cls)

            idx_forget = forget_mask.nonzero(as_tuple=True)[0]
            if idx_forget.numel():
                rand = choices[torch.randint(0, len(valid), (idx_forget.numel(),), device=device)]
                labels_train[idx_forget] = rand

            optimizer.zero_grad()

            # ---- CMF fixed head forward: logits = z @ W_fixed^T * temp
            # NOTE: keep your CMF code assumptions: extract_features + _preprocess_feats_for_cmf exist
            with torch.no_grad():
                W_fixed = model.CMFweights.weight.detach()
                temp = getattr(model.args, "temperature", 1.0)

            f = model.extract_features(inputs)
            z = model._preprocess_feats_for_cmf(f)
            logits = (z @ W_fixed.t()) * temp

            loss = F.cross_entropy(logits, labels_train)
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            # iter-eval hooks
            _maybe_iter_eval(global_iter)

            if getattr(args, "dry_run", False):
                break

        # -------------------------
        # Epoch-end: recompute CMF + eval + log epoch_end_iter/value-only
        # -------------------------
        model.eval()

        # optional: print train-set eval (same behavior as your baseline)
        test(model, device, test_loader,
             args.unlearn_class, args.class_label_names, args.num_classes,
             job_name=args.unlearn_method, set_name=f"Train Set (Epoch {epoch})")

        # key: recompute CMF on TRUE labels
        if hasattr(model, "recompute_cmf"):
            print(f"[Epoch {epoch}] Recomputing CMF with true labels...")
            model.recompute_cmf(train_loader, device=device)

        retain_acc, forget_acc = _run_forward_test(set_name=f"Test Set (Epoch {epoch})")
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

        epoch_end_iter_list.append(global_iter)
        epoch_end_retain_acc_list.append(retain_acc)   # value-only
        epoch_end_forget_acc_list.append(forget_acc)

        print(f"[Epoch {epoch}] forward: retain={retain_acc:.4f}, forget={forget_acc:.4f} (iter={global_iter})")

        if lp_every_epoch > 0 and (epoch % lp_every_epoch == 0):
            print(f"[Epoch {epoch}] Evaluating Linear Probe (epoch)...")
            lp_r, lp_f = _run_lp()
            LP_retain_acc_list.append(lp_r)
            LP_forget_acc_list.append(lp_f)
            epoch_end_LP_retain_list.append(lp_r)      # value-only
            epoch_end_LP_forget_list.append(lp_f)
            print(f"[LP@epoch{epoch}] retain={lp_r:.4f}, forget={lp_f:.4f} (iter={global_iter})")

        if ncc_every_epoch > 0 and (epoch % ncc_every_epoch == 0):
            print(f"[Epoch {epoch}] Evaluating NCC (epoch)...")
            ncc_r, mis_r, ncc_f, mis_f = _run_ncc()
            NCC_retain_acc_list.append(ncc_r)
            NCC_forget_acc_list.append(ncc_f)
            NCC_retain_mis_list.append(mis_r)
            NCC_forget_mis_list.append(mis_f)
            epoch_end_NCC_retain_list.append(ncc_r)      # value-only
            epoch_end_NCC_forget_list.append(ncc_f)
            epoch_end_NCC_retain_mis_list.append(mis_r)
            epoch_end_NCC_forget_mis_list.append(mis_f)
            print(f"[NCC@epoch{epoch}] retain={ncc_r:.4f} (mis={mis_r:.4f}) | "
                  f"forget={ncc_f:.4f} (mis={mis_f:.4f}) (iter={global_iter})")

        model.train()

    # -------------------------
    # Save logs (EXACT SAME KEYS AS YOUR BASELINE)
    # -------------------------
    model.history_log = {
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "NCC_retain_acc": NCC_retain_acc_list,
        "NCC_forget_acc": NCC_forget_acc_list,
        "NCC_retain_mis": NCC_retain_mis_list,
        "NCC_forget_mis": NCC_forget_mis_list,

        "epoch_end_iter": epoch_end_iter_list,

        "iter": iter_list,
        "iter_retain_acc": iter_retain_acc_list,
        "iter_forget_acc": iter_forget_acc_list,
        "iter_LP_retain_acc": iter_LP_retain_acc_list,
        "iter_LP_forget_acc": iter_LP_forget_acc_list,
        "iter_NCC_retain_acc": iter_NCC_retain_acc_list,
        "iter_NCC_forget_acc": iter_NCC_forget_acc_list,
        "iter_NCC_retain_mis": iter_NCC_retain_mis_list,
        "iter_NCC_forget_mis": iter_NCC_forget_mis_list,

        "epoch_end_retain_acc": epoch_end_retain_acc_list,
        "epoch_end_forget_acc": epoch_end_forget_acc_list,
        "epoch_end_LP_retain": epoch_end_LP_retain_list,
        "epoch_end_LP_forget": epoch_end_LP_forget_list,
        "epoch_end_NCC_retain": epoch_end_NCC_retain_list,
        "epoch_end_NCC_forget": epoch_end_NCC_forget_list,
        "epoch_end_NCC_retain_mis": epoch_end_NCC_retain_mis_list,
        "epoch_end_NCC_forget_mis": epoch_end_NCC_forget_mis_list,
    }

    return model


@apply_prep
def random_label_unlearn_iter_eval(args, model, device,
                                  retain_loader, forget_loader,
                                  train_loader, test_loader,
                                  optimizer, epochs, test_forget_loader, **kwargs):
    from utils import test
    from utils import get_model
    from unlearn.tools import freeze_except_last_layer, zero_except_last_layer
    import torch.optim as optim
    import evaluation
    from evaluation.nc_cmf import ncc_mismatch

    # -------------------------
    # Freeze / zero settings
    # -------------------------
    if getattr(args, "freeze_except_last", None):
        freeze_except_last_layer(model, args.freeze_except_last)
    elif getattr(args, "zero_last_layer", None):
        zero_except_last_layer(model, args.zero_last_layer)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    num_classes = args.num_classes

    # -------------------------
    # Build subsets once
    # -------------------------
    full_forget = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(full_forget), "num_forget_samples larger than forget_dataset size"
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f
    )

    full_retain = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(full_retain), "num_retain_samples larger than retain_dataset size"
    gen_r = torch.Generator().manual_seed(args.seed + 1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r
    )

    mixed_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([forget_sub, retain_sub]),
        batch_size=args.batch_size,
        shuffle=True
    )

    forget_classes = set(args.unlearn_class)
    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    # -------------------------
    # Eval frequency (iteration-based)
    # -------------------------
    eval_every_iter = getattr(args, "eval_every_iter", 0)   # forward test
    lp_every_iter   = getattr(args, "lp_every_iter", 0)     # LP
    ncc_every_iter  = getattr(args, "ncc_every_iter", 0)    # NCC

    # Keep your epoch-based LP/NCC knobs unchanged
    lp_every_epoch  = getattr(args, "lp_every", 1)   # 0 disables LP entirely (epoch-based)
    ncc_every_epoch = getattr(args, "ncc_every", 1)  # 0 disables NCC entirely (epoch-based)

    # -------------------------
    # Logs
    # -------------------------
    # iteration-based curves (dense or sparse depending on *_every_iter)
    iter_list = []
    iter_retain_acc_list, iter_forget_acc_list = [], []
    iter_LP_retain_acc_list, iter_LP_forget_acc_list = [], []
    iter_NCC_retain_acc_list, iter_NCC_forget_acc_list = [], []
    iter_NCC_retain_mis_list, iter_NCC_forget_mis_list = [], []

    # epoch-based curves (as before) + epoch_end_iter marks (KEY for plotting on iteration x-axis)
    epoch_list = list(range(0, epochs + 1))
    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    NCC_retain_acc_list, NCC_forget_acc_list = [], []
    NCC_retain_mis_list, NCC_forget_mis_list = [], []

    # NEW: map each epoch eval point to iteration index
    

    # Optional: record epoch-end LP/NCC on iter axis as (iter, value)
    epoch_end_iter_list = []  # epoch 0 baseline at iter 0
    epoch_end_forget_acc_list, epoch_end_retain_acc_list = [], []
    epoch_end_LP_retain_list, epoch_end_LP_forget_list = [], []
    epoch_end_NCC_retain_list, epoch_end_NCC_forget_list = [], []
    epoch_end_NCC_retain_mis_list, epoch_end_NCC_forget_mis_list = [], []

    # -------------------------
    # Helper evals
    # -------------------------
    def _run_forward_test(set_name: str):
        r, f, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=set_name
        )
        return r, f

    def _run_lp():
        outs_lp = evaluation.run_linear_probe_on_fresh_clone(
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
        return outs_lp["acc_test_retain"], outs_lp["acc_test_forget"]

    def _run_ncc():
        pool_mode = getattr(args, "nc_pool_mode", "avg")
        outs_retain = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_loader,
            device=device,
            pool_mode=pool_mode,
        )
        outs_forget = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_forget_loader,
            device=device,
            pool_mode=pool_mode,
        )
        return (outs_retain["ncc_acc"], outs_retain["ncc_mismatch"],
                outs_forget["ncc_acc"], outs_forget["ncc_mismatch"])

    def _maybe_iter_eval(global_iter: int):
        """
        Iteration-based evaluation hooks.
        - forward test if global_iter % eval_every_iter == 0
        - LP if global_iter % lp_every_iter == 0
        - NCC if global_iter % ncc_every_iter == 0
        """
        need = ((eval_every_iter and global_iter % eval_every_iter == 0) or
                (lp_every_iter   and global_iter % lp_every_iter   == 0) or
                (ncc_every_iter  and global_iter % ncc_every_iter  == 0))
        if not need:
            return

        model.eval()

        # --- forward test ---
        if eval_every_iter and (global_iter % eval_every_iter == 0):
            r, f = _run_forward_test(set_name=f"Test Set (Iter {global_iter})")
            iter_list.append(global_iter)
            iter_retain_acc_list.append(r)
            iter_forget_acc_list.append(f)
            print(f"[Iter {global_iter}] forward: retain={r:.4f}, forget={f:.4f}")

        # --- LP ---
        if lp_every_iter and (global_iter % lp_every_iter == 0):
            print(f"[Iter {global_iter}] Evaluating Linear Probe (iter)...")
            lp_r, lp_f = _run_lp()
            iter_LP_retain_acc_list.append(lp_r)
            iter_LP_forget_acc_list.append(lp_f)
            print(f"[LP@iter{global_iter}] retain={lp_r:.4f}, forget={lp_f:.4f}")

        # --- NCC ---
        if ncc_every_iter and (global_iter % ncc_every_iter == 0):
            print(f"[Iter {global_iter}] Evaluating NCC (iter)...")
            ncc_r, mis_r, ncc_f, mis_f = _run_ncc()
            iter_NCC_retain_acc_list.append(ncc_r)
            iter_NCC_forget_acc_list.append(ncc_f)
            iter_NCC_retain_mis_list.append(mis_r)
            iter_NCC_forget_mis_list.append(mis_f)
            print(f"[NCC@iter{global_iter}] retain={ncc_r:.4f} (mis={mis_r:.4f}) | "
                  f"forget={ncc_f:.4f} (mis={mis_f:.4f})")

        model.train()

    # -------------------------
    # Baseline @ epoch 0 (iter=0)
    # -------------------------
    model.eval()
    print("[Before Unlearning] Evaluating model (epoch 0 baseline)")
    retain_acc0, forget_acc0 = _run_forward_test(set_name="Test Set (Epoch 0)")

    epoch_end_iter_list.append(0)  # epoch 0 baseline at iter 0

    epoch_end_retain_acc_list.append((retain_acc0))
    epoch_end_forget_acc_list.append((forget_acc0))

    retain_acc_list.append(retain_acc0)
    forget_acc_list.append(forget_acc0)

    # Epoch-based LP baseline (unchanged)
    if lp_every_epoch != 0:
        print("[LP] Running linear probe at epoch 0...")
        lp_r0, lp_f0 = _run_lp()
        LP_retain_acc_list.append(lp_r0)
        LP_forget_acc_list.append(lp_f0)
        # also map to iter-axis as an epoch-end style point at iter=0
        epoch_end_LP_retain_list.append((lp_r0))
        epoch_end_LP_forget_list.append((lp_f0))
        print(f"[LP@epoch0] retain={lp_r0:.4f}, forget={lp_f0:.4f}")

    # Epoch-based NCC baseline (unchanged)
    if ncc_every_epoch != 0:
        print("[NCC] Running NCC evaluation at epoch 0...")
        ncc_r0, mis_r0, ncc_f0, mis_f0 = _run_ncc()
        NCC_retain_acc_list.append(ncc_r0)
        NCC_forget_acc_list.append(ncc_f0)
        NCC_retain_mis_list.append(mis_r0)
        NCC_forget_mis_list.append(mis_f0)
        # also map to iter-axis at iter=0
        epoch_end_NCC_retain_list.append((ncc_r0))
        epoch_end_NCC_forget_list.append((ncc_f0))
        epoch_end_NCC_retain_mis_list.append((mis_r0))
        epoch_end_NCC_forget_mis_list.append((mis_f0))
        print(f"[NCC@epoch0] retain={ncc_r0:.4f} (mis={mis_r0:.4f}) | "
              f"forget={ncc_f0:.4f} (mis={mis_f0:.4f})")

    


    model.train()

    # -------------------------
    # Training epochs + iteration eval
    # -------------------------
    global_iter = 0
    for epoch in range(1, epochs + 1):
        print(f"[Epoch {epoch}]")
        model.train()

        for inputs, labels in mixed_loader:
            global_iter += 1
            inputs, labels = inputs.to(device), labels.to(device)

            # relabel only forget classes in this batch
            forget_mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (labels == cls)

            idx_forget = forget_mask.nonzero(as_tuple=True)[0]
            if idx_forget.numel():
                rand = choices[torch.randint(0, len(valid), (idx_forget.numel(),), device=device)]
                labels[idx_forget] = rand

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.nll_loss(logits, labels)
            loss.backward()
            optimizer.step()

            # iteration-based eval hooks
            _maybe_iter_eval(global_iter)

            if args.dry_run:
                break

        # -------------------------
        # Epoch-end eval (still do output every epoch end)
        # + NEW: mark its iteration index for learning-curve x-axis
        # -------------------------
        model.eval()

        # keep your train-set test print (same behavior as original)
        test(model, device, train_loader,
             args.unlearn_class, args.class_label_names, args.num_classes,
             job_name=args.unlearn_method, set_name=f"Train Set (Epoch {epoch})")

        retain_acc, forget_acc = _run_forward_test(set_name=f"Test Set (Epoch {epoch})")
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

        # NEW: epoch-end -> iteration index
        epoch_end_iter_list.append(global_iter)

        epoch_end_retain_acc_list.append((retain_acc))
        epoch_end_forget_acc_list.append((forget_acc))

        print(f"[Epoch {epoch}] forward: retain={retain_acc:.4f}, forget={forget_acc:.4f} (iter={global_iter})")

        # Epoch-based LP/NCC (unchanged) + also record (iter, value) for plotting
        if lp_every_epoch > 0 and (epoch % lp_every_epoch == 0):
            print(f"[Epoch {epoch}] Evaluating Linear Probe (epoch)...")
            lp_r, lp_f = _run_lp()
            LP_retain_acc_list.append(lp_r)
            LP_forget_acc_list.append(lp_f)
            epoch_end_LP_retain_list.append((lp_r))
            epoch_end_LP_forget_list.append((lp_f))
            print(f"[LP@epoch{epoch}] retain={lp_r:.4f}, forget={lp_f:.4f} (iter={global_iter})")

        if ncc_every_epoch > 0 and (epoch % ncc_every_epoch == 0):
            print(f"[Epoch {epoch}] Evaluating NCC (epoch)...")
            ncc_r, mis_r, ncc_f, mis_f = _run_ncc()
            NCC_retain_acc_list.append(ncc_r)
            NCC_forget_acc_list.append(ncc_f)
            NCC_retain_mis_list.append(mis_r)
            NCC_forget_mis_list.append(mis_f)
            epoch_end_NCC_retain_list.append((ncc_r))
            epoch_end_NCC_forget_list.append(( ncc_f))
            epoch_end_NCC_retain_mis_list.append((mis_r))
            epoch_end_NCC_forget_mis_list.append(( mis_f))
            print(f"[NCC@epoch{epoch}] retain={ncc_r:.4f} (mis={mis_r:.4f}) | "
                  f"forget={ncc_f:.4f} (mis={mis_f:.4f}) (iter={global_iter})")

        model.train()

    # -------------------------
    # Save logs
    # -------------------------
    model.history_log = {
        # epoch-based (original style)
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "NCC_retain_acc": NCC_retain_acc_list,
        "NCC_forget_acc": NCC_forget_acc_list,
        "NCC_retain_mis": NCC_retain_mis_list,
        "NCC_forget_mis": NCC_forget_mis_list,

        # NEW: epoch-eval points aligned onto iteration x-axis
        "epoch_end_iter": epoch_end_iter_list,  # len = epochs+1, maps to retain_acc/forget_acc (epoch-based)

        # iteration-based eval curves (for learning curve at fine granularity)
        "iter": iter_list,
        "iter_retain_acc": iter_retain_acc_list,
        "iter_forget_acc": iter_forget_acc_list,
        "iter_LP_retain_acc": iter_LP_retain_acc_list,
        "iter_LP_forget_acc": iter_LP_forget_acc_list,
        "iter_NCC_retain_acc": iter_NCC_retain_acc_list,
        "iter_NCC_forget_acc": iter_NCC_forget_acc_list,
        "iter_NCC_retain_mis": iter_NCC_retain_mis_list,
        "iter_NCC_forget_mis": iter_NCC_forget_mis_list,

        # Optional: epoch-end LP/NCC as (iter, value) for easy scatter overlays
        "epoch_end_retain_acc": epoch_end_retain_acc_list,
        "epoch_end_forget_acc": epoch_end_forget_acc_list,
        "epoch_end_LP_retain": epoch_end_LP_retain_list,
        "epoch_end_LP_forget": epoch_end_LP_forget_list,
        "epoch_end_NCC_retain": epoch_end_NCC_retain_list,
        "epoch_end_NCC_forget": epoch_end_NCC_forget_list,
        "epoch_end_NCC_retain_mis": epoch_end_NCC_retain_mis_list,
        "epoch_end_NCC_forget_mis": epoch_end_NCC_forget_mis_list,
    }

    return model


def random_label_once(args, model, device,
                      retain_loader, forget_loader,
                      train_loader, test_loader,
                      optimizer, epochs, test_forget_loader, **kwargs):
    from unlearn.tools import maybe_eval_and_save
    # Random relabeling
    
    forget_dataset = copy.deepcopy(forget_loader.dataset)
    old_labels = np.array(forget_dataset.labels)
    valid = [c for c in range(args.num_classes) if c not in args.unlearn_class]
    rand_choice = torch.randint(0, len(valid), size=len(forget_dataset))
    new_labels = [valid[i] for i in rand_choice]
    forget_dataset.update_labels(new_labels)
    
    #forget_dataset.update_labels((forget_dataset.labels+np.random.randint(1, args.num_classes, len(forget_dataset) ))%args.num_classes)
    assert args.num_forget_samples <= len(forget_dataset), "num_forget_samples larger than forget_dataset size"
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_dataset, _ = torch.utils.data.random_split(forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples], generator=gen_f)
    
    #Subsample retain set. 
    retain_dataset = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(retain_dataset), "num_retain_samples larger than retain_dataset size"
    gen_r = torch.Generator().manual_seed(args.seed + 1)
    retain_dataset, _ = torch.utils.data.random_split(retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples], generator=gen_r)

    train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])

    mixed_loader= torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True)


    maybe_eval_and_save(0, model, args,
                        train_loader, test_loader,
                        retain_loader, forget_loader, test_forget_loader)
    model.train()
    for epoch in range(1, epochs + 1):
        for inputs, labels in mixed_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.nll_loss(logits, labels)
            loss.backward()
            optimizer.step()
            if args.dry_run:
                break
        maybe_eval_and_save(epoch, model, args,
                            train_loader, test_loader,
                            retain_loader, forget_loader, test_forget_loader)
        if args.dry_run:
            break
    return model

@apply_prep
def random_label_unlearn(args, model, device,
                         retain_loader, forget_loader,
                         train_loader, test_loader,
                         optimizer, epochs, test_forget_loader, **kwargs):
    from utils import test
    from utils import get_model
    from unlearn.tools import freeze_except_last_layer, zero_except_last_layer
    import torch.optim as optim
    import evaluation
    from evaluation.nc_cmf import ncc_mismatch
    
    if getattr(args, "freeze_except_last", None):
        freeze_except_last_layer(model, args.freeze_except_last)
    elif getattr(args, "zero_last_layer", None):
        zero_except_last_layer(model, args.zero_last_layer)

    optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    nesterov=True
    )

    num_classes = args.num_classes

    #only build the subset at beginning
    full_forget = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(full_forget), "num_forget_samples larger than forget_dataset size"
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f
    )
    full_retain = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(full_retain), "num_retain_samples larger than retain_dataset size"
    gen_r = torch.Generator().manual_seed(args.seed+1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r
    )
    mixed_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([forget_sub, retain_sub]), batch_size=args.batch_size, shuffle=True)

    forget_classes = set(args.unlearn_class)

    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)


    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    LP_history_list = []
    NCC_retain_acc_list, NCC_forget_acc_list = [], []
    NCC_retain_mis_list, NCC_forget_mis_list = [], []
    epoch_list = list(range(0, epochs + 1))

    model.eval()
    print("[Before Unlearning] Evaluating model (epoch 0 baseline)")
    retain_acc, forget_acc, _ = test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (Epoch 0)"
    )
    retain_acc_list.append(retain_acc)
    forget_acc_list.append(forget_acc)

    # Optional LP baseline at epoch 0
    lp_every = getattr(args, "lp_every", 1)  # set 0 to disable LP entirely
    if lp_every != 0:
        print("[LP] Running linear probe at epoch 0...")
        outs_LP0 = evaluation.run_linear_probe_on_fresh_clone(
            args=args,
            get_model_fn=get_model,
            device_probe=device,         # use same device; set to CPU/other GPU if desired
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
    ncc_every = getattr(args, "ncc_every", 1)  # set 0 to disable NCC entirely
    # ===== NCC @ epoch 0 =====
    if ncc_every != 0:
        print("[NCC] Running NCC evaluation at epoch 0...")
        pool_mode = getattr(args, "nc_pool_mode", "avg")

        outs_ncc0_retain = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_loader,
            device=device,
            pool_mode=pool_mode,
        )
        outs_ncc0_forget = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_forget_loader,
            device=device,
            pool_mode=pool_mode,
        )

        NCC_retain_acc_list.append(outs_ncc0_retain["ncc_acc"])
        NCC_forget_acc_list.append(outs_ncc0_forget["ncc_acc"])
        NCC_retain_mis_list.append(outs_ncc0_retain["ncc_mismatch"])
        NCC_forget_mis_list.append(outs_ncc0_forget["ncc_mismatch"])

        print(
            f"[NCC@epoch0] "
            f"retain={outs_ncc0_retain['ncc_acc']:.4f} "
            f"(mis={outs_ncc0_retain['ncc_mismatch']:.4f}) | "
            f"forget={outs_ncc0_forget['ncc_acc']:.4f} "
            f"(mis={outs_ncc0_forget['ncc_mismatch']:.4f})"
        )


    # ---------------------------
    # Training epochs
    # ---------------------------
    for epoch in range(1, epochs + 1):
        print(f"[Epoch {epoch}]")
        model.train()
        
        for inputs, labels in mixed_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            forget_mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (labels == cls)
            idx_forget = forget_mask.nonzero(as_tuple=True)[0]
            if idx_forget.numel():
                rand = choices[torch.randint(0, len(valid), (idx_forget.numel(),), device=device)]
                labels[idx_forget] = rand
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.nll_loss(logits, labels)
            loss.backward()
            optimizer.step()
            if args.dry_run:
                break
        
        # Eval after this epoch
        model.eval()
        test(model, device, train_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
        retain_acc, forget_acc, metric = test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
        
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)


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

        ncc_every = getattr(args, "ncc_every", 1)
        if ncc_every > 0 and (epoch % ncc_every == 0):
            print(f"[Epoch {epoch}] Evaluating NCC after CMF update...")
            pool_mode = getattr(args, "nc_pool_mode", "avg")

            outs_ncc_retain = ncc_mismatch(
                args=args,
                model=model,
                train_loader=train_loader,
                eval_loader=test_loader,
                device=device,
                pool_mode=pool_mode,
            )
            outs_ncc_forget = ncc_mismatch(
                args=args,
                model=model,
                train_loader=train_loader,
                eval_loader=test_forget_loader,
                device=device,
                pool_mode=pool_mode,
            )

            NCC_retain_acc_list.append(outs_ncc_retain["ncc_acc"])
            NCC_forget_acc_list.append(outs_ncc_forget["ncc_acc"])
            NCC_retain_mis_list.append(outs_ncc_retain["ncc_mismatch"])
            NCC_forget_mis_list.append(outs_ncc_forget["ncc_mismatch"])

            print(
                f"[NCC@epoch{epoch}] "
                f"retain={outs_ncc_retain['ncc_acc']:.4f} "
                f"(mis={outs_ncc_retain['ncc_mismatch']:.4f}) | "
                f"forget={outs_ncc_forget['ncc_acc']:.4f} "
                f"(mis={outs_ncc_forget['ncc_mismatch']:.4f})"
            )


        model.train()

    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        #"LP_history": LP_history_list,
        "NCC_retain_acc": NCC_retain_acc_list,
        "NCC_forget_acc": NCC_forget_acc_list,
        "NCC_retain_mis": NCC_retain_mis_list,
        "NCC_forget_mis": NCC_forget_mis_list,
        "epoch": epoch_list}
        
        
    return model




def re_train(args, model, device,
                         retain_loader, forget_loader,
                         train_loader, test_loader,
                         optimizer, epochs, test_forget_loader, **kwargs):
    from utils import test
    import torch.optim as optim
    num_classes = args.num_classes

    #only build the subset at beginning
    full_forget = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(full_forget), "num_forget_samples larger than forget_dataset size"
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f
    )
    full_retain = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(full_retain), "num_retain_samples larger than retain_dataset size"
    gen_r = torch.Generator().manual_seed(args.seed+1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r
    )
    mixed_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([forget_sub, retain_sub]), batch_size=args.batch_size, shuffle=True)

    forget_classes = set(args.unlearn_class)
    
    test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.gamma)
    model.train()
    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)
    best_score = -float('inf')
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience = getattr(args, "patience", 10)
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss= 0
        for inputs, labels in mixed_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            forget_mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (labels == cls)
            idx_forget = forget_mask.nonzero(as_tuple=True)[0]
            if idx_forget.numel():
                rand = choices[torch.randint(0, len(valid), (idx_forget.numel(),), device=device)]
                labels[idx_forget] = rand
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.nll_loss(logits, labels)
            loss.backward()
            train_loss += loss.detach().item()
            optimizer.step()
            if args.dry_run:
                break
        print(f"[Epoch {epoch}]")
        model.eval()
        # Evaluate the model on the test set
        retain_acc, forget_acc, metric = test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
        score = retain_acc * (1 - forget_acc)
        if score > best_score:
            best_score = score
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
            print(f"New best score: {score:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        model.train()
        
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()
    print("Best epoch: {best_epoch}")
    model.load_state_dict(best_model)
    return model



def train_random_label(args, model, device,
                       retain_loader,  # onlyusein maybe_eval_and_save()
                       forget_loader,  # idem
                       train_loader,   # training DataLoader
                       test_loader,
                       optimizer, epochs, **kwargs):
    """
    • directly train_loader；
    •  batch ，put label∈forget_classes 's/ofsamplethisrandomform
      otherclass (new != old)；
    • useuse；epoch evaluate + checkpoint
    """
    from unlearn.tools import maybe_eval_and_save
    from utils import test
    ce_loss = torch.nn.NLLLoss()
    forget_classes = set(args.unlearn_class)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # -------- 1. findto batch inin forget_set 's/of --------
            if forget_classes:
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for cls in forget_classes:
                    mask |= (labels == cls)

                if mask.any():
                    old = labels[mask]
                    # generateform new != old ：random 1..num_classes-1
                    shift = torch.randint(
                        1, args.num_classes, size=old.shape,
                        device=labels.device)
                    labels = labels.clone()          # notexternal
                    labels[mask] = (old + shift) % args.num_classes
                    

            # -------- 2. before /  / morenew --------
            optimizer.zero_grad()
            _, logits = model(inputs) if isinstance(model(inputs), tuple) \
                         else (None, model(inputs))
            loss = ce_loss(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        #print(f"[Epoch {epoch:3d}] mean_loss = {running_loss/len(train_loader):.4f}")
        if epoch%10 ==0:
            print(f"[Epoch {epoch:3d}]")
            test(model, device, test_loader,args.unlearn_class, args.class_label_names, args.num_classes,
                job_name = args.unlearn_method, set_name="Test Set")
            model.train()
        # -------- 3. epoch evaluate & checkpoint --------
    maybe_eval_and_save(
    epoch, model, args,
        train_loader, test_loader,
        retain_loader, forget_loader,
        step_interval=50)

    return model



@apply_prep
def random_label_CMF_unlearn(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    test_forget_loader, **kwargs
):
    """
    training：
      -  'forget classes' 's/ofsamplethislabelrandomreplaceis/beothernon-class（fake labels），
        usethissomelabelinfixed's/of CMF weightdotraining（onlymorenew encoder / youmorenew's/ofnumber）。
    CMF morenew（ epoch ）：
      - use train_loader 's/of【actually/reallabel】calluse model.recompute_cmf(...) classaveragevalueweight。
    """
    import copy, torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, ConcatDataset
    from utils import test, get_model
    import evaluation
    from evaluation.nc_cmf import ncc_mismatch


    # ========== 0) evaluate ==========
    print("[Before Unlearning] Evaluating CMF model")
    print("args.CMF_momentum =", getattr(args, "CMF_momentum", None))
    print("model.args.CMF_momentum =", getattr(model.args, "CMF_momentum", None))

    
    

    # ========== 1) optimalifydevice ==========
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )

    scheduler = None
    if getattr(args, "use_lr_decay", False):
        step_size = getattr(args, "lr_decay_step", 3)
        gamma = getattr(args, "lr_decay_gamma", 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    clip = getattr(args, "grad_norm_clip", None)

    # ========== 2) constructrandomsubset + loaddevice（and random_label_unlearn to/foralign）==========
    num_classes = args.num_classes
    forget_classes = set(args.unlearn_class)

    full_forget = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(full_forget), "num_forget_samples larger than forget_dataset size"
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget, [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f
    )

    full_retain = copy.deepcopy(retain_loader.dataset)
    assert args.num_retain_samples <= len(full_retain), "num_retain_samples larger than retain_dataset size"
    gen_r = torch.Generator().manual_seed(args.seed + 1)
    retain_sub, _ = torch.utils.data.random_split(
        full_retain, [args.num_retain_samples, len(full_retain) - args.num_retain_samples],
        generator=gen_r
    )

    mixed_loader = DataLoader(
        ConcatDataset([forget_sub, retain_sub]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(forget_loader, "num_workers", 0),
        pin_memory=True
    )

    # canrandomreplaceto's/of“haveclass”：havenon- forget class
    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    # ========== 3) unified/statisticand LP ==========
    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    LP_history_list = []
    Wn_list, Hn_list = [], []
    G_WW_list, G_HH_list, G_WH_list = [], [], []
    NCC_retain_acc_list, NCC_forget_acc_list = [], []
    NCC_retain_mis_list, NCC_forget_mis_list = [], []
    epoch_list = list(range(0, epochs + 1))

    # optional：epoch=0 's/of LP 
    model.eval()
    model.recompute_cmf(train_loader, device=device)

    lp_every = getattr(args, "lp_every", 1)  # set 0 to disable LP entirely
    if lp_every != 0:
        print("[LP] Running linear probe at epoch 0...")
        outs_LP0 = evaluation.run_linear_probe_on_fresh_clone(
                args=args,
                get_model_fn=get_model,
                device_probe=device,              # directlyusebefore device； CPU/one，selfline/executeform args.lp_device
                src_model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=getattr(args, "prob_batch_size", getattr(args, "probe_batch_size", 256)),
        )
        LP_retain_acc_list.append(outs_LP0["acc_test_retain"])
        LP_forget_acc_list.append(outs_LP0["acc_test_forget"])
        LP_history_list.append(outs_LP0.get("history"))
        print(f"[LP@epoch0] retain={outs_LP0['acc_test_retain']}, forget={outs_LP0['acc_test_forget']}")

    ncc_every = getattr(args, "ncc_every", 1)
    if ncc_every != 0:
        print("[NCC] Running NCC evaluation at epoch 0...")
        pool_mode = getattr(args, "nc_pool_mode", "avg")

        outs_ncc0_retain = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_loader,
            device=device,
            pool_mode=pool_mode,
        )
        outs_ncc0_forget = ncc_mismatch(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=test_forget_loader,
            device=device,
            pool_mode=pool_mode,
        )

        NCC_retain_acc_list.append(outs_ncc0_retain["ncc_acc"])
        NCC_forget_acc_list.append(outs_ncc0_forget["ncc_acc"])
        NCC_retain_mis_list.append(outs_ncc0_retain["ncc_mismatch"])
        NCC_forget_mis_list.append(outs_ncc0_forget["ncc_mismatch"])

        print(
            f"[NCC@epoch0] "
            f"retain={outs_ncc0_retain['ncc_acc']:.4f} "
            f"(mis={outs_ncc0_retain['ncc_mismatch']:.4f}) | "
            f"forget={outs_ncc0_forget['ncc_acc']:.4f} "
            f"(mis={outs_ncc0_forget['ncc_mismatch']:.4f})"
        )

    retain_acc, forget_acc, metric = test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set"
    )

    retain_acc_list.append(retain_acc)
    forget_acc_list.append(forget_acc)

    # ========== 4) training ==========
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}]")
        model.train()

        for inputs, labels_true in mixed_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_true = labels_true.to(device, non_blocking=True)

            # ---- 4.1 construct fake labels（onlyto/for forget classsamplethiswrite）----
            labels_train = labels_true.clone()
            # bool ：inoneclass
            forget_mask = torch.zeros_like(labels_true, dtype=torch.bool)
            for cls in forget_classes:
                forget_mask |= (labels_true == cls)

            if forget_mask.any():
                n_forget = int(forget_mask.sum().item())
                #  valid classinaveragerandomget
                rand_targets = choices[torch.randint(
                    low=0, high=len(valid), size=(n_forget,), device=device
                )]
                labels_train[forget_mask] = rand_targets

            optimizer.zero_grad()

            # ---- 4.2 usefixed's/of CMF weightinbeforecompute logits + CE(fake labels) ----
            with torch.no_grad():
                W_fixed = model.CMFweights.weight.detach()  # notin batch morenew W
                temp = getattr(model.args, "temperature", 1.0)

            f = model.extract_features(inputs)
            z = model._preprocess_feats_for_cmf(f)
            logits = (z @ W_fixed.t()) * temp

            loss = F.cross_entropy(logits, labels_train)
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            if getattr(args, "dry_run", False):
                break  # onlyuseinvalidation

        # ---- 4.3 evaluate + useactually/reallabel CMF classifier ----
        model.eval()
        # key：thisinuse train_loader 's/of【actually/reallabel】intoline/executeclassaveragevalue/
        Wn, Hn, G_WW, G_HH, G_WH = model.recompute_cmf(train_loader, device=device)
        Wn_list.append(Wn)
        Hn_list.append(Hn)
        G_WW_list.append(G_WW)
        G_HH_list.append(G_HH)
        G_WH_list.append(G_WH)

        print(f"[Epoch {epoch}] Evaluating after CMF update...")
        retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

        # ---- 4.4 optional：LP（featureiscanproperty）----
        lp_every = getattr(args, "lp_every", 1)   # 0 can
        if lp_every > 0 and (epoch % lp_every == 0):
            print(f"[Epoch {epoch}] Evaluating Linear Probe after update...")
            outs_LP = evaluation.run_linear_probe_on_fresh_clone(
                args=args,
                get_model_fn=get_model,
                device_probe=device,   #  CPU/one GPU，canis/be torch.device(getattr(args, "lp_device", "cpu"))
                src_model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=getattr(args, "prob_batch_size", getattr(args, "probe_batch_size", 256)),
            )
            LP_retain_acc_list.append(outs_LP["acc_test_retain"])
            LP_forget_acc_list.append(outs_LP["acc_test_forget"])
            LP_history_list.append(outs_LP.get("history"))
            print(f"[LP@epoch{epoch}] retain={outs_LP['acc_test_retain']}, forget={outs_LP['acc_test_forget']}")
            print("epochs_trained", outs_LP.get("epochs_trained"))

        ncc_every = getattr(args, "ncc_every", 1)
        if ncc_every > 0 and (epoch % ncc_every == 0):
            print(f"[Epoch {epoch}] Evaluating NCC after CMF update...")
            pool_mode = getattr(args, "nc_pool_mode", "avg")

            outs_ncc_retain = ncc_mismatch(
                args=args,
                model=model,
                train_loader=train_loader,
                eval_loader=test_loader,
                device=device,
                pool_mode=pool_mode,
            )
            outs_ncc_forget = ncc_mismatch(
                args=args,
                model=model,
                train_loader=train_loader,
                eval_loader=test_forget_loader,
                device=device,
                pool_mode=pool_mode,
            )

            NCC_retain_acc_list.append(outs_ncc_retain["ncc_acc"])
            NCC_forget_acc_list.append(outs_ncc_forget["ncc_acc"])
            NCC_retain_mis_list.append(outs_ncc_retain["ncc_mismatch"])
            NCC_forget_mis_list.append(outs_ncc_forget["ncc_mismatch"])

            print(
                f"[NCC@epoch{epoch}] "
                f"retain={outs_ncc_retain['ncc_acc']:.4f} "
                f"(mis={outs_ncc_retain['ncc_mismatch']:.4f}) | "
                f"forget={outs_ncc_forget['ncc_acc']:.4f} "
                f"(mis={outs_ncc_forget['ncc_mismatch']:.4f})"
            )
        if scheduler is not None:
            scheduler.step()
            print(f"[Scheduler] LR after epoch {epoch}: {scheduler.get_last_lr()[0]:.6f}")

        model.train()

    # ========== 5) record ==========
    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        #"LP_history": LP_history_list,
        #"Wn": Wn_list,
        #"Hn": Hn_list,
        #"G_WW": G_WW_list,
        #"G_HH": G_HH_list,
        #"G_WH": G_WH_list,
        "NCC_retain_acc": NCC_retain_acc_list,
        "NCC_forget_acc": NCC_forget_acc_list,
        "NCC_retain_mis": NCC_retain_mis_list,
        
        "epoch": epoch_list,
    }

    return model

@apply_prep
def random_label_once_CMF_unlearn(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    test_forget_loader, **kwargs
):
    """
    random_label_once + CMF this：

    - in unlearning openbefore，onlyto/for forget classes 's/ofsamplethisrandomform retain classlabel（change all at once）；
    - newlabelonefixednotetcinoriginallabel（is/beonlynon- forget class valid insample）；
    - relabelaftertrainingpassinlabelkeepsupportnot；
    -  epoch afteruse train_loader 's/ofactually/reallabel CMF classifier；
    - other（LP / history_log）allandoriginal random_label_CMF_unlearn keepsupportconsistent。
    """
    import copy, torch, numpy as np
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, ConcatDataset
    from utils import test, get_model
    import evaluation

    # ========== 0) evaluate ==========
    print("[Before Unlearning] Evaluating CMF model (random_label_once + CMF)")
    print("args.CMF_momentum =", getattr(args, "CMF_momentum", None))

    # ========== 1) optimalifydevice ==========
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )

    scheduler = None
    if getattr(args, "use_lr_decay", False):
        step_size = getattr(args, "lr_decay_step", 3)
        gamma = getattr(args, "lr_decay_gamma", 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    clip = getattr(args, "grad_norm_clip", None)

    # ========== 2) construct forget / retain subset + random relabeling all at once forget ==========
    num_classes = args.num_classes
    forget_classes = set(args.unlearn_class)

    # ---- 2.1  forget dataset（passis UpdateableSubset）----
    full_forget = copy.deepcopy(forget_loader.dataset)
    assert args.num_forget_samples <= len(full_forget), \
        "num_forget_samples larger than forget_dataset size"

    # full_forget need labels and indices，come update_labels
    if not hasattr(full_forget, "labels"):
        raise ValueError("forget_loader.dataset have `labels` property")
    if not hasattr(full_forget, "indices"):
        raise ValueError("forget_loader.dataset have `indices` property（Subset ）")

    labels_all = np.array(full_forget.labels).astype(int)   # layernumberaccording's/oflabel，for exampledegree 50000
    indices = np.array(full_forget.indices)                 # this Subset actually/realcontain's/ofsamplethis，for exampledegree 5000

    # subset_labels isbefore forget_dataset (Subset) in's/oflabel，degree == len(indices)
    subset_labels = labels_all[indices]

    # onlyto/for subset inin forget_classes 's/ofsamplethisrelabel
    valid = [c for c in range(num_classes) if c not in forget_classes]
    if len(valid) == 0:
        raise ValueError("No valid (non-forget) classes for random relabeling.")

    forget_mask_subset = np.isin(subset_labels, list(forget_classes))
    forget_pos = np.where(forget_mask_subset)[0]   # thissomeis subset 's/ofposition 0..len(subset)-1

    print(f"[RandomLabel-Once] #forget samples in forget_dataset subset: {len(forget_pos)}")

    subset_new = subset_labels.copy()
    if len(forget_pos) > 0:
        gen = torch.Generator().manual_seed(args.seed + 2)
        rand_choice = torch.randint(
            low=0,
            high=len(valid),
            size=(len(forget_pos),),
            generator=gen
        )
        for k, pos in enumerate(forget_pos):
            # originallabelin forget_classes in，newlabel valid get，ensure != originallabel
            subset_new[pos] = int(valid[rand_choice[k]])

        if not hasattr(full_forget, "update_labels"):
            raise ValueError("forget_loader.dataset actually/real update_labels(new_labels)")

        # ⚠️ thisininput's/of new_labels degree == len(indices)，
        # willin update_labels in labels_hold[self.indices] = new_labels 's/ofstylewritelayerlabelnumbergroup
        full_forget.update_labels(subset_new.tolist())
    else:
        print("[RandomLabel-Once] Warning: no forget-class samples found in this subset.")

    # ---- 2.2 inrelabel's/of full_forget extract from forget_sub（andoriginalkeepsupportconsistent）----
    gen_f = torch.Generator().manual_seed(args.seed)
    forget_sub, _ = torch.utils.data.random_split(
        full_forget,
        [args.num_forget_samples, len(full_forget) - args.num_forget_samples],
        generator=gen_f
    )

    # ---- 2.3 retain_sub keepsupportnot ----
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

    # ========== 3) unified/statisticand LP ==========
    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    LP_history_list = []
    Wn_list, Hn_list = [], []
    G_WW_list, G_HH_list, G_WH_list = [], [], []
    epoch_list = list(range(0, epochs + 1))

    # epoch=0：firstuseactually/reallabelonce CMF，againdo LP + test
    model.eval()
    model.recompute_cmf(train_loader, device=device)

    lp_every = getattr(args, "lp_every", 1)  # set 0 to disable LP entirely
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

    retain_acc, forget_acc, metric = test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set"
    )
    retain_acc_list.append(retain_acc)
    forget_acc_list.append(forget_acc)

    # ========== 4) training ==========
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}]")
        model.train()

        for inputs, labels in mixed_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)  # thisin's/of labels isconsistent's/of“newlabel”

            optimizer.zero_grad()

            # useusefixed's/of CMF weight
            with torch.no_grad():
                W_fixed = model.CMFweights.weight.detach()
                temp = getattr(model.args, "temperature", 1.0)

            f = model.extract_features(inputs)
            z = model._preprocess_feats_for_cmf(f)
            logits = (z @ W_fixed.t()) * temp

            loss = F.cross_entropy(logits, labels)
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            if getattr(args, "dry_run", False):
                break

        # ---- 4.3 CMF morenew + test ----
        model.eval()
        Wn, Hn, G_WW, G_HH, G_WH = model.recompute_cmf(train_loader, device=device)
        Wn_list.append(Wn)
        Hn_list.append(Hn)
        G_WW_list.append(G_WW)
        G_HH_list.append(G_HH)
        G_WH_list.append(G_WH)

        print(f"[Epoch {epoch}] Evaluating after CMF update...")
        retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

        # ---- 4.4 LP ----
        lp_every = getattr(args, "lp_every", 1)
        if lp_every > 0 and (epoch % lp_every == 0):
            print(f"[Epoch {epoch}] Evaluating Linear Probe after update...")
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

        if scheduler is not None:
            scheduler.step()
            print(f"[Scheduler] LR after epoch {epoch}: "
                  f"{scheduler.get_last_lr()[0]:.6f}")

        model.train()

    # ========== 5) record ==========
    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        # "LP_history": LP_history_list,
        # "Wn": Wn_list,
        # "Hn": Hn_list,
        # "G_WW": G_WW_list,
        # "G_HH": G_HH_list,
        # "G_WH": G_WH_list,
        "epoch": epoch_list,
    }

    return model
