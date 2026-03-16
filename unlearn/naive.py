import torch
import torch.nn.functional as F
from unlearn.tools import  apply_prep
import evaluation
import pandas as pd
import time


@apply_prep
def unlearn_naive(args, model, device, retain_loader, forget_loader, train_loader, test_loader, optimizer, epochs, test_forget_loader, **kwargs):
    from utils import test

    import torch.optim as optim
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    method = args.unlearn_method
    clip = args.grad_norm_clip
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(
        forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples]
    )
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples]
    )
    naive_retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    
    test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")

    model.train()
    forget_iterator = iter(forget_loader)
    retain_acc_list, forget_acc_list = [], []
    epoch_list = list(range(1, epochs + 1))
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}:")
        for data, target in naive_retain_loader:
            optimizer.zero_grad()
            if "ascent" in method:
                try:
                    data_f, target_f = next(forget_iterator)
                except:
                    forget_iterator = iter(forget_loader)
                    data_f, target_f = next(forget_iterator)
                data_f, target_f = data_f.to(device), target_f.to(device)
                output = model(data_f)
                loss = -1.0 * F.cross_entropy(output, target_f)
                loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if "descent" in method:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
            optimizer.step()
            if args.dry_run:
                break
        print("  Evaluating after update...") 
        retain_acc, forget_acc, metric = test(model, device, test_loader,
         args.unlearn_class, args.class_label_names, args.num_classes,
         job_name=args.unlearn_method, set_name="Test Set")
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)
        model.train()
    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "epoch": epoch_list}
    print({
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "epoch": epoch_list})
    print(model.history_log)
    return model





import torch
import torch.nn.functional as F
from unlearn.tools import apply_prep

@apply_prep
def unlearn_naive_CMF_old(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    test_forget_loader, **kwargs
):
    from utils import test
    from utils import get_model

    print("[Before Unlearning] Evaluating CMF model")
    print("args.CMF_momentum =", getattr(args, "CMF_momentum", None))
    print("model.args.CMF_momentum =", getattr(model.args, "CMF_momentum", None))


    test(
        model, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set"
    )

    model.train()

    # Setup optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # Optional: clip gradient norm
    clip = args.grad_norm_clip

    # Use only a subset for forgetting and retaining
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(
        forget_dataset, [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples]
    )
    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset, [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples]
    )
    naive_retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    forget_iterator = iter(forget_loader)

    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list = [], []
    LP_history_list = []
    epoch_list = list(range(1, epochs + 1))
    for epoch in range(1, epochs + 1):
        for x, y in naive_retain_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Negative gradient for forgetting data
            if "ascent" in args.unlearn_method:
                try:
                    x_f, y_f = next(forget_iterator)
                except StopIteration:
                    forget_iterator = iter(forget_loader)
                    x_f, y_f = next(forget_iterator)
                x_f, y_f = x_f.to(device), y_f.to(device)
                loss_f, _ = model.forward_a((x_f, y_f), stage="train")
                (-loss_f).backward()

            # Positive gradient for retaining data
            if "descent" in args.unlearn_method:
                loss_r, _ = model.forward_a((x, y), stage="train")
                loss_r.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        model.eval()
        print(f"[Epoch {epoch}] Evaluating after update...")
        retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)
        
        
        lp_every = getattr(args, "lp_every", 1)   # =1；is/be0canLP
        do_lp = (lp_every > 0) and (epoch % lp_every == 0)

        if do_lp:
            print(f"[Epoch {epoch}] Evaluating Linear Probe after update...")
            # recommended：device_probe = torch.device("cpu") ；use GPU，givetoone， cuda:1
            #device_probe = torch.device("cpu")

            outs_LP = evaluation.run_linear_probe_on_fresh_clone(
                args=args,
                get_model_fn=get_model,           # youhave's/ofworknumber
                device_probe=device,       # training LP 's/of
                src_model=model,                  # intraining's/ofmodel
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=args.prob_batch_size,
            )

            LP_retain_acc_list.append(outs_LP["acc_test_retain"])
            LP_forget_acc_list.append(outs_LP["acc_test_forget"])
            LP_history_list.append(outs_LP.get("history"))
            print(f"[LP@epoch{epoch}] retain={outs_LP['acc_test_retain']}, forget={outs_LP['acc_test_forget']}")
        
        model.train()

    model.history_log = {
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "LP_retain_acc": LP_retain_acc_list,
        "LP_forget_acc": LP_forget_acc_list,
        "LP_history": LP_history_list,
        "epoch": epoch_list}
    #print({"retain_acc": retain_acc_list, "forget_acc": forget_acc_list,"epoch": epoch_list})
    #print(model.history_log)
    
    return model




from unlearn.tools import apply_prep
import torch
import torch.nn.functional as F

@apply_prep
def unlearn_naive_CMF(
    args, model, device,
    retain_loader, forget_loader,
    train_loader, test_loader,
    optimizer, epochs,
    test_forget_loader, **kwargs
):
    """
    Naive unlearning with CMF:
    - Align evaluation geometry at baseline (eval() + recompute_cmf()).
    - Record baseline classifier and LP performance at epoch 0.
    - Training with ascent/descent losses on forget/retain subsets.
    - At the end of each epoch, recompute CMF geometry and evaluate again.
    """

    from utils import test, get_model

    print("[Before Unlearning] Evaluating CMF model")
    print("args.CMF_momentum =", getattr(args, "CMF_momentum", None))
    print("model.args.CMF_momentum =", getattr(model.args, "CMF_momentum", None))

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )
    clip = getattr(args, "grad_norm_clip", None)

    # -------------------------
    # Create forget/retain subsets
    # -------------------------
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(
        forget_dataset,
        [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples]
    )
    naive_forget_loader = torch.utils.data.DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    forget_iterator = iter(naive_forget_loader)

    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset,
        [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples]
    )
    naive_retain_loader = torch.utils.data.DataLoader(
        retain_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True,
        num_workers=getattr(forget_loader, "num_workers", 2),
    )

    # -------------------------
    # Hyperparameters for loss
    # -------------------------
    lam        = getattr(args, "align_coef",       0.0)
    alpha      = getattr(args, "forget_scale",     0.5)
    use_margin = getattr(args, "use_margin_forget", False)
    margin     = getattr(args, "forget_margin",    0.0)
    beta_m     = getattr(args, "beta_margin",      0.0)

    # -------------------------
    # Logs (start from epoch 0)
    # -------------------------
    retain_acc_list, forget_acc_list = [], []
    LP_retain_acc_list, LP_forget_acc_list, LP_history_list = [], [], []
    G_WW_list, G_HH_list, G_WH_list = [], [], []
    epoch_list = [0]

    # =========================================================
    # Baseline: eval()  + test() + LP
    # =========================================================
    model.eval()
    model.recompute_cmf(train_loader, device=device)

    # Classifier (CMF head) baseline
    with torch.no_grad():
        retain_acc, forget_acc, _ = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name="Test Set (Epoch 0)"
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

    # Linear Probe baseline
    lp_every = getattr(args, "lp_every", 1)  # set 0 to disable LP entirely
    if lp_every != 0:
        print("[LP] Running linear probe at epoch 0...")
        bs_probe = getattr(args, "prob_batch_size", getattr(args, "probe_batch_size", 256))
        outs_LP = evaluation.run_linear_probe_on_fresh_clone(
            args=args,
            get_model_fn=get_model,
            device_probe=device,
            src_model=model,
            train_loader=train_loader,   # must use full training set
            test_loader=test_loader,
            num_classes=args.num_classes,
            bs_probe=bs_probe,
        )
        LP_retain_acc_list.append(outs_LP["acc_test_retain"])
        LP_forget_acc_list.append(outs_LP["acc_test_forget"])
        LP_history_list.append(outs_LP.get("history"))
        print(f"[LP@epoch0] retain={outs_LP['acc_test_retain']}, forget={outs_LP['acc_test_forget']}")

    model.train()

    # =========================================================
    # Main training loop
    # =========================================================
    for epoch in range(1, epochs + 1):
        print("device:", device)
        print("model device:", next(model.parameters()).device)
        
        epoch_start = time.time()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for i, (x,y) in enumerate(naive_retain_loader):
            
            x, y = x.to(device), y.to(device)

            starter.record()
            optimizer.zero_grad()
            
            #print("Get fixed CMF weights")
            with torch.no_grad():
                W_fixed = model.CMFweights.weight.detach()

            total_loss_item = 0.0

            # Forgetting (ascent)
            if "ascent" in args.unlearn_method:
                #print("Start forget batch")
                try:
                    x_f, y_f = next(forget_iterator)
                except StopIteration:
                    forget_iterator = iter(naive_forget_loader)
                    x_f, y_f = next(forget_iterator)
                x_f, y_f = x_f.to(device), y_f.to(device)
                #print("Extract features for forget batch")
                f_f = model.extract_features(x_f)
                z_f = model._preprocess_feats_for_cmf(f_f)
                logits_f = (z_f @ W_fixed.t()) * getattr(model.args, "temperature", 1.0)
                #print("Compute losses for forget batch")
                L_ce_f    = F.cross_entropy(logits_f, y_f)
                L_align_f = 1.0 - (z_f * W_fixed[y_f]).sum(1).mean()

                if use_margin:
                    s_pos = logits_f.gather(1, y_f[:, None]).squeeze(1)
                    s_mask = torch.zeros_like(logits_f)
                    s_mask.scatter_(1, y_f[:, None], 1e9)
                    s_top_other, _ = (logits_f - s_mask).max(dim=1)
                    L_margin = F.relu(margin + s_pos - s_top_other).mean()
                else:
                    L_margin = 0.0

                #total_loss = total_loss + (-(alpha * L_ce_f)) + lam * L_align_f + beta_m * L_margin
                loss_ascent = (-(alpha * L_ce_f)) + lam * L_align_f + beta_m * L_margin
                loss_ascent.backward()                 # thisin backward
                total_loss_item += float(loss_ascent.detach())

            # Retaining (descent)
            if "descent" in args.unlearn_method:
                #print("Start forget batch ")
                #print("Extract features for retain batch")
                f_r = model.extract_features(x)
                z_r = model._preprocess_feats_for_cmf(f_r)
                logits_r = (z_r @ W_fixed.t()) * getattr(model.args, "temperature", 1.0)
                #print
                L_ce_r    = F.cross_entropy(logits_r, y)
                L_align_r = 1.0 - (z_r * W_fixed[y]).sum(1).mean()

                #total_loss = total_loss + L_ce_r + lam * L_align_r

                loss_descent = L_ce_r + lam * L_align_r
                loss_descent.backward()                # ★key：twotime backward
                total_loss_item += float(loss_descent.detach())

            
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            ender.record()
            if i % 50 == 0:
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)  # GPU compute ms
                print(f"i={i} loss={total_loss_item:.4f} gpu_step_ms={ms:.1f}")

        epoch_end = time.time()
        print(f"[Epoch {epoch}] wall time: {epoch_end - epoch_start:.1f}s")


        # End-of-epoch evaluation
        # -------------------------
        model.eval()
        Wn, Hn, G_WW, G_HH, G_WH = model.recompute_cmf(train_loader, device=device)
        #G_WW_list.append(G_WW); G_HH_list.append(G_HH); G_WH_list.append(G_WH)

        # Classifier evaluation\
        with torch.no_grad():
            retain_acc, forget_acc, _ = test(
                model, device, test_loader,
                args.unlearn_class, args.class_label_names, args.num_classes,
                job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
            )
            retain_acc_list.append(retain_acc)
            forget_acc_list.append(forget_acc)

        # Linear Probe evaluation
        lp_every = getattr(args, "lp_every", 1)
        if lp_every > 0 and (epoch % lp_every == 0):
            outs_LP = evaluation.run_linear_probe_on_fresh_clone(
                args=args,
                get_model_fn=get_model,
                device_probe=device,
                src_model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=args.num_classes,
                bs_probe=bs_probe,
            )
            LP_retain_acc_list.append(outs_LP["acc_test_retain"])
            LP_forget_acc_list.append(outs_LP["acc_test_forget"])
            LP_history_list.append(outs_LP.get("history"))
            print(f"[LP@epoch{epoch}] retain={outs_LP['acc_test_retain']}, forget={outs_LP['acc_test_forget']}")

        epoch_list.append(epoch)
        model.train()

    # -------------------------
    # Save history
    # -------------------------
    model.history_log = {
        "retain_acc":     retain_acc_list,
        "forget_acc":     forget_acc_list,
        "LP_retain_acc":  LP_retain_acc_list,
        "LP_forget_acc":  LP_forget_acc_list,
        "epoch":          epoch_list,
    }

    return model
