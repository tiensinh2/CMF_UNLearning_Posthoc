# evaluation/linear_prob_whold.py
# -*- coding: utf-8 -*-
"""
ResNet-18 linear probing (whole pipeline):
- Per-layer hooking (conv1 + layer{1..4}.{0,1}.bn2), collect features
- Last-layer probing via linear_prob.linear_probe_last_layer
- Combine results, sort and plot
"""

from typing import List, Dict, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# Import last-layer LP from your previous file
from evaluation.linear_prob import linear_probe_last_layer, _train_linear_probe


# ----------------------------- helpers -----------------------------

def _is_cmf(args) -> bool:
    return ("CMF" in getattr(args, "unlearn_method", ""))


def _backbone(model: nn.Module, args) -> nn.Module:
    """Return the backbone to hook into."""
    return model.encoder if _is_cmf(args) else model


def _pool(out: torch.Tensor) -> torch.Tensor:
    """Global average pool 4D features to [B, C]."""
    if out.dim() == 4:
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
    return out


def _resnet18_bn2_layers() -> List[str]:
    return [
        "conv1",
        "layer1.0.bn2", "layer1.1.bn2",
        "layer2.0.bn2", "layer2.1.bn2",
        "layer3.0.bn2", "layer3.1.bn2",
        "layer4.0.bn2", "layer4.1.bn2",
    ]


@torch.no_grad()
def _collect_labels_only(loader) -> np.ndarray:
    ys = []
    for _xb, yb in loader:
        ys.append(yb.cpu().numpy())
    return np.concatenate(ys, axis=0)


def _split_masks(labels_np: np.ndarray, forget_classes: List[int]):
    fm = np.isin(labels_np, forget_classes)
    rm = ~fm
    return rm, fm


def _train_linear_head(Xtr: torch.Tensor, ytr: torch.Tensor, num_classes: int,
                       device, lr=0.05, momentum=0.9, weight_decay=0.0,
                       bs=64, max_epochs=150, patience=30, seed=0) -> Tuple[nn.Module, float]:
    torch.manual_seed(seed)
    C = Xtr.size(1)
    clf = nn.Linear(C, num_classes).to(device)
    opt = optim.SGD(clf.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(Xtr, ytr)
    g = torch.Generator().manual_seed(seed)
    ld = DataLoader(ds, batch_size=bs, shuffle=True, generator=g)

    best = 0.0
    no_imp = 0
    for _ in range(max_epochs):
        clf.train()
        for bx, by in ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = loss_fn(clf(bx), by)
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            pr = clf(Xtr).argmax(1)
            acc = (pr.cpu() == ytr.cpu()).float().mean().item()
        if acc > best:
            best, no_imp = acc, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    return clf, best


def _eval_split(pred: np.ndarray, y: np.ndarray, mask: np.ndarray):
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return None
    return float((pred[idx] == y[idx]).mean())


# --------------------- per-layer hooking (no fc) ---------------------

def linear_probe_all_layers_per_layerhook(
    args, model, train_loader, test_loader,
    num_classes, bs_probe, device='cpu'
):
    model.eval()
    bb = _backbone(model, args)
    layer_names = _resnet18_bn2_layers()

    results = []

    for name in layer_names:
        print(f"[LP] probing layer: {name}")
        feats_tr, labels_tr_list = [], []
        feats_te, labels_te_list = [], []

        mod = dict(bb.named_modules())[name]

        # ---- TRAIN pass:  feats and labels（ensureconsistent）----
        def hook_train(_m, _i, out):
            feats_tr.append(_pool(out).detach().cpu())

        h = mod.register_forward_hook(hook_train)
        with torch.no_grad():
            for xb, yb in train_loader:
                _ = model(xb.to(device))
                labels_tr_list.append(yb.detach().cpu())
        h.remove()

        Xtr = torch.cat(feats_tr, dim=0)
        ytr = torch.cat(labels_tr_list, dim=0).long()   # <-- andfeatureoneto

        # ---- TEST pass:  feats and labels（ensureconsistent）----
        def hook_test(_m, _i, out):
            feats_te.append(_pool(out).detach().cpu())

        h = mod.register_forward_hook(hook_test)
        with torch.no_grad():
            for xb, yb in test_loader:
                _ = model(xb.to(device))
                labels_te_list.append(yb.detach().cpu())
        h.remove()

        Xte = torch.cat(feats_te, dim=0)
        yte = torch.cat(labels_te_list, dim=0).long()

        # ---- afterandyouin's/oftraining/evaluateconsistent ----
        Xtr_t = Xtr.float().to(device)
        Xte_t = Xte.float().to(device)
        ytr_t = ytr.to(device)

        clf, best_val_acc, best_epoch, history = _train_linear_probe(
        Xtr_t, ytr_t, num_classes,
        device=device,
        seed=getattr(args, 'seed', 42),
        lr=getattr(args, 'lr', 0.1),
        momentum=getattr(args, 'momentum', 0.9),
        weight_decay=getattr(args, 'weight_decay', 0.0),
        batch_size=bs_probe,
        patience=50,
        max_epochs=200,
        val_ratio=0.1,   # oryou's/of
        )

        clf.eval()
        with torch.no_grad():
            pr_tr = clf(Xtr_t).argmax(1).cpu().numpy()
            pr_te = clf(Xte_t).argmax(1).cpu().numpy()

        lab_tr = ytr.cpu().numpy()
        lab_te = yte.cpu().numpy()
        rm_tr, fm_tr = _split_masks(lab_tr, args.unlearn_class)
        rm_te, fm_te = _split_masks(lab_te, args.unlearn_class)

        results.append({
            'layer': name,
            'acc_train_retain': _eval_split(pr_tr, lab_tr, rm_tr),
            'acc_train_forget': _eval_split(pr_tr, lab_tr, fm_tr),
            'acc_test_retain':  _eval_split(pr_te, lab_te, rm_te),
            'acc_test_forget':  _eval_split(pr_te, lab_te, fm_te),
            'best_train_acc':   float(best_val_acc),
        })

    return results



# ------------------ combine, sort and plot ------------------

def block_sort_key(name: str):
    if name == "conv1":
        return (0, 0)
    for layer_idx in range(1, 5):
        prefix = f"layer{layer_idx}."
        if name.startswith(prefix):
            parts = name.split('.')
            try:
                block_idx = int(parts[1])
            except Exception:
                block_idx = 0
            return (layer_idx, block_idx)
    if name == "penultimate":
        return (5, -1)
    if name == "fc":
        return (5, 0)
    return (6, 0)


def plot_LP(args, out_df: pd.DataFrame):
    layers = sorted(out_df['layer'].tolist(), key=block_sort_key)

    def take(col):
        return [out_df.loc[out_df['layer'] == l, col].values[0] for l in layers]

    acc_ret_train = take('acc_train_retain')
    acc_for_train = take('acc_train_forget')
    acc_ret_test  = take('acc_test_retain')
    acc_for_test  = take('acc_test_forget')

    out_dir = f"result/plot_layer_LP/{args.exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/{args.arch}_{','.join([str(v) for v in args.unlearn_class])}.png"

    plt.figure(figsize=(12, 6))
    #plt.plot(layers, acc_ret_train, marker='o', label='Train Retain')
    #plt.plot(layers, acc_for_train, marker='s', label='Train Forget')
    plt.plot(layers, acc_ret_test,  marker='^', label='Test Retain')
    plt.plot(layers, acc_for_test,  marker='v', label='Test Forget')
    plt.xticks(rotation=90)
    plt.ylabel('Accuracy')
    plt.title(f"{args.exp_name} | {args.dataset} | {args.arch} | forget={','.join([str(v) for v in args.unlearn_class])}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[LP] saved plot: {out_file}")


def run_linear_probe_both(
    args, model, train_loader, test_loader,
    device, num_classes, bs_probe
) -> Tuple[List[Dict], pd.DataFrame]:
    outs_all = linear_probe_all_layers_per_layerhook(
        args, model, train_loader, test_loader,
        num_classes=num_classes, bs_probe=bs_probe, device=device
    )

    # Last layer (penultimate)
    last_res = linear_probe_last_layer(
        args, model, train_loader, test_loader,
        num_classes=num_classes, bs_probe=bs_probe, device=device
    )
    last_res['layer'] = 'penultimate'

    lp_list = outs_all + [last_res]
    lp_df = pd.DataFrame(lp_list)
    return lp_list, lp_df
