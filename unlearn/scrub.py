"""SCRUB unlearning method extracted from class_forgetting-master/main.py.

This file only contains the SCRUB method wrapper (scrub_unlearn) and its direct
dependencies/imports. Third-party SCRUB implementation under `scrub_thirdparty/`
is used as-is.

Usage (example):
    from scrub_method import scrub_unlearn
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.optim as optim
from torch import nn

# Third-party (kept unchanged)
from scrub_thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from scrub_thirdparty.repdistiller.distiller_zoo import DistillKL
from scrub_thirdparty.repdistiller.helper.loops import train_distill


class CMFHeadWrapper(nn.Module):
    """Wrap a CMF model so forward() returns CMF-head logits (not log-softmax)."""

    def __init__(self, cmf_model: nn.Module, detach_W: bool = True):
        super().__init__()
        self.m = cmf_model
        self.detach_W = detach_W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        f = self.m.extract_features(x)
        z = self.m._preprocess_feats_for_cmf(f)

        # CMF weights
        W = self.m.CMFweights.weight
        if self.detach_W:
            W = W.detach()

        temp = getattr(getattr(self.m, "args", None), "temperature", 1.0)
        return (z @ W.t()) * temp

    def recompute_cmf(self, *args, **kwargs):
        return self.m.recompute_cmf(*args, **kwargs)

    def unwrap_backbone(self):
        mm = self.m
        if hasattr(mm, "unwrap_backbone"):
            return mm.unwrap_backbone()
        return mm


def _subsample_loader(
    base_loader,
    num_samples: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    ds = base_loader.dataset
    if num_samples > len(ds):
        raise ValueError(f"num_samples={num_samples} > len(dataset)={len(ds)}")

    gen = torch.Generator().manual_seed(seed)
    sub, _ = torch.utils.data.random_split(ds, [num_samples, len(ds) - num_samples], generator=gen)

    return torch.utils.data.DataLoader(
        sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def scrub_CMF_unlearn(
    args: Any,
    model: nn.Module,
    device: torch.device,
    retain_loader,
    forget_loader,
    train_loader=None,
    test_loader=None,
    **kwargs,
) -> nn.Module:
    """
    SCRUB with CMF head.

    Training:
      - repdistiller distillation operates on CMF-head logits via CMFHeadWrapper.
      - teacher uses detached W (fixed CMF head).
      - student optionally detaches W in forward (default True) to make encoder-only updates stable.

    CMF recompute:
      - Recompute CMF on TRUE labels at end of *every* epoch (mandatory).

    Evaluation:
      - Always evaluate model_s_raw (raw CMF model), after recompute_cmf().

    Returns:
      - model_s_raw (raw CMF model), so your pipeline can keep using CMFweights/recompute/etc.
    """
    assert train_loader is not None, "SCRUB-CMF requires train_loader to recompute CMF weights."
    assert test_loader is not None, "SCRUB-CMF requires test_loader for evaluation."

    from utils import test

    # -------------------------
    # Subsample forget/retain for SCRUB
    # -------------------------
    seed = int(getattr(args, "seed", 0))
    nw_f = getattr(forget_loader, "num_workers", 0)
    nw_r = getattr(retain_loader, "num_workers", 0)

    scrub_forget_loader = _subsample_loader(
        forget_loader,
        num_samples=int(args.num_forget_samples),
        batch_size=int(args.scrub_del_bsz),
        seed=seed + 123,
        num_workers=nw_f,
        pin_memory=True,
    )
    scrub_retain_loader = _subsample_loader(
        retain_loader,
        num_samples=int(args.num_retain_samples),
        batch_size=int(args.scrub_sgda_bsz),
        seed=seed + 456,
        num_workers=nw_r,
        pin_memory=True,
    )

    # -------------------------
    # SCRUB/SGDA hyperparams (mirror original code)
    # -------------------------
    args.optim = "sgd"
    args.gamma = 0.99
    args.alpha = 0.001
    args.beta = 0
    args.smoothing = 0.0
    args.msteps = int(args.scrub_msteps)
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4
    args.distill = "kd"

    args.sgda_epochs = int(args.scrub_epochs)
    args.sgda_learning_rate = float(args.lr)
    args.lr_decay_epochs = [3, 5, 9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9

    # -------------------------
    # Teacher / Student (raw CMF models)
    # -------------------------
    model_t_raw = copy.deepcopy(model)
    model_s_raw = copy.deepcopy(model)

    # Put raw models on device
    model_t_raw.to(device)
    model_s_raw.to(device)

    # Teacher frozen (saves memory + matches KD assumption)
    model_t_raw.eval()
    for p in model_t_raw.parameters():
        p.requires_grad_(False)

    # Align CMF weights to TRUE labels before training
    model_t_raw.recompute_cmf(train_loader, device=device)
    model_s_raw.eval()
    model_s_raw.recompute_cmf(train_loader, device=device)

    # Wrapper to expose CMF logits to repdistiller
    detach_W_student = bool(getattr(args, "scrub_cmf_detach_W", True))
    model_t = CMFHeadWrapper(model_t_raw, detach_W=True)  # teacher always fixed W
    model_s = CMFHeadWrapper(model_s_raw, detach_W=detach_W_student)

    # -------------------------
    # SWA (as in original SCRUB code)
    # -------------------------
    beta = 0.1

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return (1 - beta) * averaged_model_parameter + beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)

    module_list = nn.ModuleList([model_s, model_t])
    trainable_list = nn.ModuleList([model_s])

    # -------------------------
    # Losses (repdistiller)
    # -------------------------
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)
    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    # -------------------------
    # Optimizer
    # -------------------------
    if args.optim == "sgd":
        optimizer = optim.SGD(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            momentum=args.sgda_momentum,
            weight_decay=args.sgda_weight_decay,
        )
    elif args.optim == "adam":
        optimizer = optim.Adam(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            weight_decay=args.sgda_weight_decay,
        )
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            momentum=args.sgda_momentum,
            weight_decay=args.sgda_weight_decay,
        )
    else:
        raise ValueError(f"Unsupported args.optim: {args.optim}")

    # -------------------------
    # Move wrappers/loss/SWA to device
    # -------------------------
    module_list.to(device)
    criterion_list.to(device)
    swa_model.to(device)

    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True

    # -------------------------
    # Per-epoch eval logs
    # -------------------------
    retain_acc_list, forget_acc_list, epoch_list = [], [], []

    # Optional: epoch 0 baseline (after initial CMF recompute)
    model_s_raw.eval()
    r0, f0, _ = test(
        model_s_raw,
        device,
        test_loader,
        args.unlearn_class,
        args.class_label_names,
        args.num_classes,
        job_name=args.unlearn_method,
        set_name="Test Set (Epoch 0)",
    )
    retain_acc_list.append(r0)
    forget_acc_list.append(f0)
    epoch_list.append(0)

    # -------------------------
    # SGDA loop
    # -------------------------
    model_s.train()
    for epoch in range(1, args.sgda_epochs + 1):
        _lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        # Maximize on forget (diverge from teacher in CMF-logit space)
        max_loss = 0.0
        if epoch <= args.msteps:
            max_loss = train_distill(
                epoch,
                scrub_forget_loader,
                module_list,
                swa_model,
                criterion_list,
                optimizer,
                args,
                "maximize",
            )

        # Minimize on retain (CE + KL in CMF-logit space)
        min_acc, min_loss = train_distill(
            epoch,
            scrub_retain_loader,
            module_list,
            swa_model,
            criterion_list,
            optimizer,
            args,
            "minimize",
        )

        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)

        # =========================
        # MANDATORY: recompute CMF every epoch (TRUE labels)
        # =========================
        model_s_raw.eval()
        model_s_raw.recompute_cmf(train_loader, device=device)

        # Evaluate using RAW CMF model (canonical pipeline)
        r, f, _ = test(
            model_s_raw,
            device,
            test_loader,
            args.unlearn_class,
            args.class_label_names,
            args.num_classes,
            job_name=args.unlearn_method,
            set_name=f"Test Set (Epoch {epoch})",
        )
        retain_acc_list.append(r)
        forget_acc_list.append(f)
        epoch_list.append(epoch)

        model_s.train()

    # -------------------------
    # Attach history_log (for your main.py json dump)
    # -------------------------
    model_s_raw.history_log = {
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
        "scrub_msteps": int(args.msteps),
        "scrub_epochs": int(args.sgda_epochs),
        "scrub_del_bsz": int(args.scrub_del_bsz),
        "scrub_sgda_bsz": int(args.scrub_sgda_bsz),
        "lr": float(args.sgda_learning_rate),
        "kd_T": float(args.kd_T),
        # optional training stats (last epoch values)
        "last_maximize_loss": float(max_loss) if isinstance(max_loss, (float, int)) else None,
        "last_minimize_loss": float(min_loss) if isinstance(min_loss, (float, int)) else None,
        "last_minimize_acc": float(min_acc) if isinstance(min_acc, (float, int)) else None,
    }

    return model_s_raw



def scrub_unlearn(
    args: Any,
    model: nn.Module,
    device: torch.device,
    retain_loader,
    forget_loader,
    train_loader=None,
    test_loader=None,
    **kwargs,
):
    """
    SCRUB unlearning (as in class_forgetting-master/main.py).

    Notes:
      - Subsamples retain/forget datasets to args.num_retain_samples / args.num_forget_samples.
      - Performs an initial "maximize" phase on forget data for `args.scrub_msteps` epochs,
        then "minimize" on retain data for `args.scrub_epochs` epochs (SGDA style).
      - Uses repdistiller's `train_distill` loop and `DistillKL` loss.

    Required args fields (expected in `args`):
      - num_forget_samples, num_retain_samples
      - scrub_del_bsz, scrub_sgda_bsz
      - scrub_msteps, scrub_epochs
      - lr  (used as sgda_learning_rate)
    """
    from utils import test, get_model
    # ---- Subsample datasets ----
    forget_dataset = forget_loader.dataset
    forget_dataset, _ = torch.utils.data.random_split(
        forget_dataset,
        [args.num_forget_samples, len(forget_dataset) - args.num_forget_samples],
    )
    scrub_forget_loader = torch.utils.data.DataLoader(
        forget_dataset, batch_size=args.scrub_del_bsz, shuffle=True
    )

    retain_dataset = retain_loader.dataset
    retain_dataset, _ = torch.utils.data.random_split(
        retain_dataset,
        [args.num_retain_samples, len(retain_dataset) - args.num_retain_samples],
    )
    scrub_retain_loader = torch.utils.data.DataLoader(
        retain_dataset, batch_size=args.scrub_sgda_bsz, shuffle=True
    )

    # ---- SCRUB/SGDA hyperparams (mirrors original code) ----
    args.optim = "sgd"
    args.gamma = 0.99
    args.alpha = 0.001
    args.beta = 0
    args.smoothing = 0.0
    args.msteps = args.scrub_msteps
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4
    args.distill = "kd"

    args.sgda_epochs = args.scrub_epochs
    args.sgda_learning_rate = args.lr
    args.lr_decay_epochs = [3, 5, 9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9

    # ---- Teacher / student / SWA ----
    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)
    model_t.eval()
    for p in model_t.parameters():
        p.requires_grad_(False)
    del model  
    beta = 0.1

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return (1 - beta) * averaged_model_parameter + beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)

    module_list = nn.ModuleList([model_s, model_t])
    trainable_list = nn.ModuleList([model_s])

    # ---- Losses (repdistiller) ----
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    # ---- Optimizer ----
    if args.optim == "sgd":
        optimizer = optim.SGD(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            momentum=args.sgda_momentum,
            weight_decay=args.sgda_weight_decay,
        )
    elif args.optim == "adam":
        optimizer = optim.Adam(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            weight_decay=args.sgda_weight_decay,
        )
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            momentum=args.sgda_momentum,
            weight_decay=args.sgda_weight_decay,
        )
    else:
        raise ValueError(f"Unsupported args.optim: {args.optim}")

    # ---- Device ----
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        swa_model.cuda()

    

    retain_acc_list, forget_acc_list = [], []
    epoch_list = [0] + list(range(1, args.sgda_epochs + 1))
    print("[Before Unlearning] Evaluating (epoch 0 baseline)")
    retain_acc, forget_acc, _ = test(
        model_s, device, test_loader,
        args.unlearn_class, args.class_label_names, args.num_classes,
        job_name=args.unlearn_method, set_name="Test Set (Epoch 0)"
    )
    retain_acc_list.append(retain_acc)
    forget_acc_list.append(forget_acc)

    model_s.train()

    model_s.eval()
    # ---- SGDA loop ----
    for epoch in range(1, args.sgda_epochs + 1):
        _lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        maximize_loss = 0.0
        if epoch <= args.msteps:
            maximize_loss = train_distill(
                epoch, scrub_forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize"
            )

        train_acc, train_loss = train_distill(
            epoch, scrub_retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize"
        )

        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)

         # ----- Eval after this epoch -----
        model_s.eval()

        retain_acc, forget_acc, _ = test(
            model_s, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set (Epoch {epoch})"
        )
        retain_acc_list.append(retain_acc)
        forget_acc_list.append(forget_acc)

        model_s.train()
    model_s.history_log = {
        "epoch": epoch_list,
        "retain_acc": retain_acc_list,
        "forget_acc": forget_acc_list,
    }

    return model_s
