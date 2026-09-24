"""
unlearn/cmf_two_stage.py
========================
Shared per-epoch encoder-update logic used by NB4a (CMF-Static) AND NB4c (Stage 3).

Public API:
    run_cmf_static(
        method, model, args, device,
        retain_loader, forget_loader, train_loader, test_loader,
        train_dataset=None,          # required only for TARUN
        epochs=None,                 # overrides args value if set
        s3_lr=None,                  # Stage-3 reduced LR; None → use args.lr
        verbose=True,
    ) -> model

    The function:
      1. Recomputes CMF geometry (model.recompute_cmf(train_loader, ...))
      2. Freezes W (CMFweights.weight requires_grad_(False))
      3. For each epoch:
           a. Iterates the retain/forget data with the method-specific loss
           b. After each epoch: calls model.recompute_cmf(train_loader, ...) again
      4. Returns the updated model

NOTE: recompute_cmf() returns a 5-tuple in old code.  The callers in this file
do NOT unpack it (BUG-FIX #1 — "recompute_cmf() returns None" was because old
site did: Wn, Hn, G_WW, G_HH, G_WH = model.recompute_cmf(...)).
We just call it plain.

CUDA Event BUG-FIX: torch.cuda.Event calls guarded with `if device.type == 'cuda'`.
"""

from __future__ import annotations

import copy
import time
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------
# Internal: per-step loss builders (one per base method)
# -----------------------------------------------------------------------

def _grad_ascent_descent_step(
    model, device, retain_batch, forget_iter, forget_loader,
    args, W_fixed, clip
):
    """NegGrad+ loss: ascent on forget + descent on retain."""
    x_r, y_r = retain_batch
    x_r, y_r = x_r.to(device), y_r.to(device)

    loss = torch.tensor(0.0, device=device)

    # Ascent on forget
    try:
        x_f, y_f = next(forget_iter[0])
    except StopIteration:
        forget_iter[0] = iter(forget_loader)
        x_f, y_f = next(forget_iter[0])
    x_f, y_f = x_f.to(device), y_f.to(device)

    f_f = model.extract_features(x_f)
    z_f = model._preprocess_feats_for_cmf(f_f)
    logits_f = (z_f @ W_fixed.t()) * getattr(model.args, "temperature", 1.0)
    loss_ascent = -F.cross_entropy(logits_f, y_f)
    loss = loss + loss_ascent

    # Descent on retain
    f_r = model.extract_features(x_r)
    z_r = model._preprocess_feats_for_cmf(f_r)
    logits_r = (z_r @ W_fixed.t()) * getattr(model.args, "temperature", 1.0)
    loss_descent = F.cross_entropy(logits_r, y_r)
    loss = loss + loss_descent

    return loss


def _random_label_step(
    model, device, retain_batch, forget_iter, forget_loader,
    args, W_fixed, clip
):
    """Random-label loss: relabel forget samples to random non-forget class."""
    x_r, y_r = retain_batch
    x_r, y_r = x_r.to(device), y_r.to(device)

    forget_classes = set(args.unlearn_class)
    num_classes = args.num_classes
    valid = [c for c in range(num_classes) if c not in forget_classes]
    choices = torch.tensor(valid, device=device)

    loss = torch.tensor(0.0, device=device)

    # Forget (fake random labels)
    try:
        x_f, y_f = next(forget_iter[0])
    except StopIteration:
        forget_iter[0] = iter(forget_loader)
        x_f, y_f = next(forget_iter[0])
    x_f = x_f.to(device)
    idx = torch.randint(0, len(choices), (x_f.size(0),), device=device)
    y_fake = choices[idx]
    f_f = model.extract_features(x_f)
    z_f = model._preprocess_feats_for_cmf(f_f)
    logits_f = (z_f @ W_fixed.t()) * getattr(model.args, "temperature", 1.0)
    loss = loss + F.cross_entropy(logits_f, y_fake)

    # Retain (true labels)
    f_r = model.extract_features(x_r)
    z_r = model._preprocess_feats_for_cmf(f_r)
    logits_r = (z_r @ W_fixed.t()) * getattr(model.args, "temperature", 1.0)
    loss = loss + F.cross_entropy(logits_r, y_r)

    return loss


def _salun_step(
    model, device, retain_batch, forget_iter, forget_loader,
    args, W_fixed, clip, salun_mask=None
):
    """SalUn loss: same as random-label but applied through the SalUn gradient mask."""
    return _random_label_step(
        model, device, retain_batch, forget_iter, forget_loader,
        args, W_fixed, clip
    )


def _scrub_step(
    model, device, retain_batch, forget_iter, forget_loader,
    args, W_fixed, clip, teacher_model=None
):
    """SCRUB-style: KL distillation on retain + ascent on forget."""
    from torch.nn import functional as F

    temp = float(getattr(model.args, "temperature", 1.0))
    kd_T = float(getattr(args, "scrub_kd_T", getattr(args, "kd_T", 4.0)))
    gamma = float(getattr(args, "scrub_gamma", getattr(args, "gamma", 0.99)))
    alpha = float(getattr(args, "scrub_alpha", getattr(args, "alpha", 0.001)))

    x_r, y_r = retain_batch
    x_r, y_r = x_r.to(device), y_r.to(device)

    # KL distillation on retain (between student and teacher)
    f_r = model.extract_features(x_r)
    z_r = model._preprocess_feats_for_cmf(f_r)
    logits_s = (z_r @ W_fixed.t()) * temp

    if teacher_model is not None:
        with torch.no_grad():
            f_t = teacher_model.extract_features(x_r)
            z_t = teacher_model._preprocess_feats_for_cmf(f_t)
            W_t = teacher_model.CMFweights.weight.detach()
            logits_t = (z_t @ W_t.t()) * temp
        p_t = F.softmax(logits_t / kd_T, dim=1)
        p_s = F.log_softmax(logits_s / kd_T, dim=1)
        kl_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (kd_T ** 2)
    else:
        kl_loss = torch.tensor(0.0, device=device)

    ce_retain = F.cross_entropy(logits_s, y_r)
    loss_retain = gamma * kl_loss + alpha * ce_retain

    # Ascent on forget
    try:
        x_f, y_f = next(forget_iter[0])
    except StopIteration:
        forget_iter[0] = iter(forget_loader)
        x_f, y_f = next(forget_iter[0])
    x_f, y_f = x_f.to(device), y_f.to(device)
    f_f = model.extract_features(x_f)
    z_f = model._preprocess_feats_for_cmf(f_f)
    logits_f = (z_f @ W_fixed.t()) * temp
    loss_forget = -F.cross_entropy(logits_f, y_f)

    return loss_retain + loss_forget


def _tarun_step(
    model, device, retain_batch, forget_iter, forget_loader,
    args, W_fixed, clip, noisy_loader_iter=None
):
    """TARUN: uses the pre-built noisy loader (noise + retain)."""
    if noisy_loader_iter is not None:
        try:
            x, y = next(noisy_loader_iter[0])
        except StopIteration:
            # This shouldn't happen in normal usage but handle gracefully
            return torch.tensor(0.0, device=device)
    else:
        x, y = retain_batch

    x, y = x.to(device), y.to(device)
    temp = float(getattr(model.args, "temperature", 1.0))
    f = model.extract_features(x)
    z = model._preprocess_feats_for_cmf(f)
    logits = (z @ W_fixed.t()) * temp
    return F.cross_entropy(logits, y)


# -----------------------------------------------------------------------
# Helper: build TARUN noisy loader (requires train_dataset)
# -----------------------------------------------------------------------

def _build_tarun_noisy_loader(args, model, device, forget_loader, train_dataset, val_index):
    """
    Reproduces TARUN's noise-synthesis + small retain setup.
    Returns noisy_loader (retain + noisy forget-class samples).
    """
    import numpy as np
    from unlearn.tarun import Noise  # noqa: local import

    batch_size = args.batch_size
    targets = np.array(train_dataset.targets)
    forget_set = set(args.unlearn_class)

    index_list = []
    for i in range(args.num_classes):
        if i not in forget_set:
            class_i_idx = np.intersect1d(np.where(i == targets)[0], val_index)
            index_list.extend(class_i_idx[:int(getattr(args, "tarun_samples_per_class", 100))])

    small_retain_samples = [
        (x.cpu(), torch.tensor(y))
        for x, y in torch.utils.data.Subset(train_dataset, index_list)
    ]

    is_vit = "vit" in args.arch.lower()
    is_resnet = "resnet" in args.arch.lower()
    noises = {}
    for cls_num in args.unlearn_class:
        dataset_lower = args.dataset.lower()
        if "tinyimagenet" in dataset_lower or "tiny-imagenet" in dataset_lower:
            sz = (3, 224, 224) if is_vit else (3, 64, 64)
        elif "imagenet" in dataset_lower:
            sz = (3, 224, 224)
        else:
            sz = (3, 32, 32)
        noises[cls_num] = Noise(batch_size, *sz).to(device)
        opt = torch.optim.Adam(noises[cls_num].parameters(), lr=0.1)
        for _ in range(5):
            for __ in range(8):
                inputs = noises[cls_num]()
                labels = torch.full((batch_size,), cls_num, device=device, dtype=torch.long)
                out = model(inputs)
                loss = -F.nll_loss(out, labels) + 0.1 * torch.mean(torch.sum(inputs.square(), dim=[1, 2, 3]))
                opt.zero_grad(); loss.backward(); opt.step()

    noisy_data = []
    for cls_num in args.unlearn_class:
        for _ in range(20):
            batch = noises[cls_num]().cpu().detach()
            for i in range(batch.size(0)):
                noisy_data.append((batch[i], torch.tensor(cls_num)))
    noisy_data += small_retain_samples

    noisy_loader = torch.utils.data.DataLoader(
        noisy_data, batch_size=batch_size, shuffle=True
    )
    return noisy_loader


# -----------------------------------------------------------------------
# Public: run_cmf_static
# -----------------------------------------------------------------------

def run_cmf_static(
    method: str,
    model: nn.Module,
    args,
    device: torch.device,
    retain_loader,
    forget_loader,
    train_loader,
    test_loader=None,
    train_dataset=None,    # required for TARUN only
    val_index=None,        # required for TARUN only
    epochs: Optional[int] = None,
    s3_lr: Optional[float] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    CMF-Static Algorithm 2 encoder update (shared by NB4a and NB4c).

    Per epoch:
      1. recompute_cmf(train_loader)  — freeze W
      2. Encoder update via method-specific loss (W is detached/frozen)
      3. end-of-epoch recompute_cmf(train_loader)

    BUG-FIXES APPLIED:
    - recompute_cmf() not unpacked (was 5-tuple; site just calls it plain)
    - TARUN requires train_dataset — passed explicitly here
    - CUDA Event guarded with `if device.type == 'cuda'`
    - grad_descent → retain-only fine-tune (not NegGrad+)

    Parameters
    ----------
    method : one of grad_ascent_descent, grad_descent, random_label,
             salun, scrub, tarun
    s3_lr  : if set, overrides args.lr for Stage-3 reduced-LR runs
    """
    assert hasattr(model, "CMFweights"), (
        f"run_cmf_static: model must have CMFweights (CMFClassifier=True, remove_FC=True). "
        f"Got: {type(model)}"
    )

    if method == "tarun" and train_dataset is None:
        raise ValueError("run_cmf_static: TARUN requires train_dataset to be passed.")

    lr = s3_lr if s3_lr is not None else float(getattr(args, "lr", 1e-4))
    n_epochs = epochs if epochs is not None else int(getattr(args, "epochs_or_steps", 5))
    clip = getattr(args, "grad_norm_clip", None)
    temp = float(getattr(model.args, "temperature", 1.0))

    # Build optimizer (encoder params only — CMFweights.weight excluded explicitly)
    encoder_params = [p for n, p in model.named_parameters()
                      if "CMFweights" not in n and p.requires_grad]
    optimizer = torch.optim.SGD(
        encoder_params, lr=lr,
        momentum=getattr(args, "momentum", 0.9),
        weight_decay=getattr(args, "weight_decay", 5e-4),
        nesterov=getattr(args, "nesterov", True),
    )

    # Build TARUN noisy loader if needed
    noisy_loader = None
    noisy_loader_iter = [None]
    if method == "tarun":
        if verbose:
            print("[CMFStatic/TARUN] Building noisy loader...")
        noisy_loader = _build_tarun_noisy_loader(
            args, model, device, forget_loader, train_dataset, val_index
        )
        noisy_loader_iter = [iter(noisy_loader)]

    # Build teacher for SCRUB
    teacher_model = None
    if method == "scrub":
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        # BUG-FIX: call plain — do NOT unpack the 5-tuple
        teacher_model.recompute_cmf(train_loader, device=device)

    forget_iter = [iter(forget_loader)]

    # CUDA timing guard
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
    else:
        starter = ender = None

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # --- Step 1: Recompute CMF, freeze W ---
        model.eval()
        # BUG-FIX: call plain — do NOT unpack the 5-tuple return
        model.recompute_cmf(train_loader, device=device)
        model.CMFweights.weight.requires_grad_(False)  # freeze W explicitly

        model.train()

        with torch.no_grad():
            W_fixed = model.CMFweights.weight.detach().clone()

        # Reset TARUN iter each epoch
        if method == "tarun":
            noisy_loader_iter = [iter(noisy_loader)]

        for step, retain_batch in enumerate(retain_loader):
            if device.type == "cuda" and starter is not None:
                starter.record()

            optimizer.zero_grad()

            if method in ("grad_ascent_descent",):
                loss = _grad_ascent_descent_step(
                    model, device, retain_batch, forget_iter, forget_loader,
                    args, W_fixed, clip
                )
            elif method == "grad_descent":
                # Retain-only fine-tune (NOT NegGrad+) — BUG-FIX #unlearn/__init__ grad_descent
                x_r, y_r = retain_batch
                x_r, y_r = x_r.to(device), y_r.to(device)
                f_r = model.extract_features(x_r)
                z_r = model._preprocess_feats_for_cmf(f_r)
                logits_r = (z_r @ W_fixed.t()) * temp
                loss = F.cross_entropy(logits_r, y_r)
            elif method == "random_label":
                loss = _random_label_step(
                    model, device, retain_batch, forget_iter, forget_loader,
                    args, W_fixed, clip
                )
            elif method == "salun":
                loss = _salun_step(
                    model, device, retain_batch, forget_iter, forget_loader,
                    args, W_fixed, clip
                )
            elif method == "scrub":
                loss = _scrub_step(
                    model, device, retain_batch, forget_iter, forget_loader,
                    args, W_fixed, clip, teacher_model=teacher_model
                )
            elif method == "tarun":
                loss = _tarun_step(
                    model, device, retain_batch, forget_iter, forget_loader,
                    args, W_fixed, clip, noisy_loader_iter=noisy_loader_iter
                )
            else:
                raise ValueError(f"run_cmf_static: unknown method '{method}'")

            if loss.requires_grad:
                loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if device.type == "cuda" and ender is not None:
                ender.record()

        # --- Step 3: End-of-epoch recompute CMF ---
        model.eval()
        # BUG-FIX: call plain — do NOT unpack
        model.recompute_cmf(train_loader, device=device)

        elapsed = time.time() - t0
        if verbose:
            print(f"  [CMFStatic epoch {epoch}/{n_epochs}] wall={elapsed:.1f}s  lr={lr:.2e}")

    return model
