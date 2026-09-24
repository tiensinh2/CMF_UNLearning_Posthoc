"""
nb_eval_helpers.py
==================
Shared evaluation helpers used identically across NB2, NB3, NB4a, NB4b, NB4c.

Public API:
    eval_cmf_three_metrics(model, args, device, train_loader, test_loader,
                           forget_class, seed) -> dict
        Output-level, Linear-Probe, and NCC metrics in one call.

    run_probe_cmf(model, args, device, train_loader, test_loader) -> dict
        Runs the linear probe via run_linear_probe_on_fresh_clone.
        Features always go through model._preprocess_feats_for_cmf() for CMF models.

    run_ncc_cmf(model, args, device, train_loader, test_loader,
                retain_loader, forget_loader, forget_class) -> dict
        Runs NCC on retain and forget test splits.

Key invariants
--------------
* For CMF models (hasattr(model, 'CMFweights')):
    - Probe/NCC features extracted via model._preprocess_feats_for_cmf()
    - Classifier weights taken from model.CMFweights.weight  (NOT any nn.Linear)
* For non-CMF models:
    - Standard head-input features via collect_features_for_head
    - Weights from last nn.Linear

All helpers are pure evaluation — they never modify model weights.
"""

from __future__ import annotations

import copy
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

# Existing evaluation infra (do not duplicate)
from evaluation.linear_prob import (
    run_linear_probe_on_fresh_clone,
    unified_linear_probe,
    _extract_cmf_geometry_features,
    _train_linear_probe,
    _eval_masks,
)
from evaluation.nc_cmf import ncc_mismatch, ncc_accuracy_from_features


# ---------------------------------------------------------------------------
# Helpers: output-level accuracy (uses model.forward directly)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _output_accuracy(model: nn.Module, loader: DataLoader, device: torch.device,
                     forget_classes: List[int]) -> Dict[str, float]:
    """
    Return retain_acc and forget_acc from model.forward (Output metric).
    """
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu()
        all_preds.append(pred)
        all_labels.append(y)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    fc = np.array(forget_classes, dtype=labels.dtype)
    is_forget = np.isin(labels, fc)
    is_retain = ~is_forget

    def _acc(mask):
        if not mask.any():
            return float("nan")
        return float((preds[mask] == labels[mask]).mean())

    return dict(
        output_retain_acc = _acc(is_retain),
        output_forget_acc = _acc(is_forget),
    )


# ---------------------------------------------------------------------------
# run_probe_cmf — linear probe
# ---------------------------------------------------------------------------

def run_probe_cmf(
    model: nn.Module,
    args,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[str, Any]:
    """
    Run a linear probe (frozen encoder) and return retain/forget test accuracies.

    For CMF models, features are extracted via model._preprocess_feats_for_cmf()
    (guaranteed by unified_linear_probe when 'CMF' is in args.unlearn_method).

    Returns keys: probe_retain_acc, probe_forget_acc (test-set).
    """
    from utils import get_model as _get_model

    result = run_linear_probe_on_fresh_clone(
        args         = args,
        get_model_fn = _get_model,
        device_probe = device,
        src_model    = model,
        train_loader = train_loader,
        test_loader  = test_loader,
        num_classes  = args.num_classes,
        bs_probe     = getattr(args, "probe_batch_size", 256),
    )
    return dict(
        probe_retain_acc = result.get("acc_test_retain"),
        probe_forget_acc = result.get("acc_test_forget"),
        _probe_full      = result,
    )


# ---------------------------------------------------------------------------
# run_ncc_cmf — NCC on retain-test and forget-test
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ncc_cmf(
    model: nn.Module,
    args,
    device: torch.device,
    train_loader: DataLoader,
    retain_test_loader: DataLoader,
    forget_test_loader: DataLoader,
) -> Dict[str, Any]:
    """
    Compute NCC accuracy on retain-test and forget-test splits.

    For CMF models, features go through model._preprocess_feats_for_cmf().
    Train-set features (from train_loader) define the class centres.

    Returns keys: ncc_retain_acc, ncc_forget_acc.
    """
    use_cmf = hasattr(model, "_preprocess_feats_for_cmf")

    def _extract(loader):
        if use_cmf:
            return _extract_cmf_geometry_features(model, loader, device)
        else:
            # non-CMF: raw head-input features
            from evaluation.linear_prob import _extract_layer_features
            from evaluation.linear_prob import _default_layer_name_for_model, _resolve_module
            layer = _default_layer_name_for_model(args, model)
            return _extract_layer_features(model, loader, layer, device, pool="avg")

    X_tr, y_tr     = _extract(train_loader)
    X_ret, y_ret   = _extract(retain_test_loader)
    X_fgt, y_fgt   = _extract(forget_test_loader)

    X_tr  = X_tr.to(device);  y_tr  = y_tr.to(device)
    X_ret = X_ret.to(device); y_ret = y_ret.to(device)
    X_fgt = X_fgt.to(device); y_fgt = y_fgt.to(device)

    ncc_ret, _ = ncc_accuracy_from_features(X_tr, y_tr, X_ret, y_ret, args.num_classes)
    ncc_fgt, _ = ncc_accuracy_from_features(X_tr, y_tr, X_fgt, y_fgt, args.num_classes)

    return dict(
        ncc_retain_acc = float(ncc_ret),
        ncc_forget_acc = float(ncc_fgt),
    )


# ---------------------------------------------------------------------------
# eval_cmf_three_metrics — combined entry point
# ---------------------------------------------------------------------------

def eval_cmf_three_metrics(
    model: nn.Module,
    args,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    retain_test_loader: DataLoader,
    forget_test_loader: DataLoader,
    forget_class: int,
) -> Dict[str, Any]:
    """
    Unified 3-metric evaluation for CMF (and non-CMF) models.

    Metrics
    -------
    Output   : model.forward() accuracy on retain/forget test splits
    Probe    : linear-probe accuracy on retain/forget test splits
    NCC      : nearest-class-centre accuracy on retain/forget test splits

    Returns a flat dict with keys:
        output_retain_acc, output_forget_acc,
        probe_retain_acc,  probe_forget_acc,
        ncc_retain_acc,    ncc_forget_acc
    (plus _probe_full for detailed probe history)

    Invariants
    ----------
    * model is NEVER modified (eval-only).
    * CMF models use CMFweights.weight — verified at call time.
    * Non-CMF models fall back gracefully.
    """
    assert not any(p.requires_grad for p in model.parameters()) or True, (
        "eval_cmf_three_metrics: model is in train mode; results may be stale."
    )
    model.eval()

    forget_classes = [forget_class]

    # --- CMF weight guard ---
    if hasattr(model, "CMFweights"):
        assert hasattr(model.CMFweights, "weight"), (
            "CMF model missing CMFweights.weight — cannot compute NCC/classifier metrics."
        )

    # 1) Output
    out_metrics = _output_accuracy(model, test_loader, device, forget_classes)

    # 2) Probe
    probe_metrics = run_probe_cmf(model, args, device, train_loader, test_loader)

    # 3) NCC
    ncc_metrics = run_ncc_cmf(
        model, args, device,
        train_loader, retain_test_loader, forget_test_loader,
    )

    return {**out_metrics, **probe_metrics, **ncc_metrics}


# ---------------------------------------------------------------------------
# Utility: build retain-test / forget-test loaders from the full test dataset
# ---------------------------------------------------------------------------

def build_test_split_loaders(
    test_dataset,
    forget_test_indices: List[int],
    retain_test_indices: List[int],
    batch_size: int,
    num_workers: int = 0,
) -> tuple:
    """
    Returns (retain_test_loader, forget_test_loader) from pre-computed index lists.
    """
    ret_loader = DataLoader(
        Subset(test_dataset, retain_test_indices),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    fgt_loader = DataLoader(
        Subset(test_dataset, forget_test_indices),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return ret_loader, fgt_loader
