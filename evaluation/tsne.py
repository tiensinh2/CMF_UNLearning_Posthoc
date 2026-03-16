# evaluation/tsne.py
# -*- coding: utf-8 -*-
"""
t-SNE canifyworknumber（ main）。

in evaluation.py inthissamplecalluse：
    from evaluation.tsne import tsne_visualize
    tsne_visualize(args, model, test_loader, device,
                   out_root="result/plot_T_SNE", also_logits=True)
"""
import os
import random
from typing import Tuple, Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .collect_feature import collect_features_for_head  # usehavenumber


# -------------------------- randomproperty --------------------------
def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------- numberaccordingand --------------------------
def _subsample(X: np.ndarray, y: np.ndarray, max_points: int = 10000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """samplethisrandomsample，add t-SNE。"""
    n = X.shape[0]
    if n <= max_points:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return X[idx], y[idx]


def compute_tsne(
    X: np.ndarray,
    *,
    use_pca: bool = True,
    pca_dim: int = 50,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """
    optional PCA again t-SNE；Returntwodimensioninput Z: [N, 2]
    """
    X = X.astype(np.float32, copy=False)
    if use_pca and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
        verbose=1,
    )
    Z = tsne.fit_transform(X)
    return Z


def plot_scatter_2d(
    Z: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: Optional[str] = None,
    *,
    forget_classes=None,
    alpha: float = 0.7,
    s: int = 6,
) -> None:
    """byclass's/of 2D dot；extract out_path rule/thenkeepmaintain PNG。"""
    plt.figure(figsize=(7.2, 6.4))
    y = np.asarray(y)
    classes = np.unique(y)

    retain_classes = [c for c in classes if c not in forget_classes]
    forget_classes = [c for c in classes if c in forget_classes]

    RETAIN_PALETTE = [
        "#1f77b4",  # C0
        "#ff7f0e",  # C1
        "#2ca02c",  # C2
        # skip C3 "#d62728" (red)
        "#9467bd",  # C4
        "#8c564b",  # C5
        "#e377c2",  # C6 (pink)
        "#7f7f7f",  # C7
        "#bcbd22",  # C8
        "#17becf",  # C9
    ]



    for i, c in enumerate(retain_classes):
        mask = (y == c)
        color = RETAIN_PALETTE[i % len(RETAIN_PALETTE)]
        #plt.scatter(Z[mask, 0], Z[mask, 1], s=s, alpha=alpha, label=str(c), c=color)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=s, alpha=alpha, c=color)

        
    for c in forget_classes:
        mask = (y == c)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=s, alpha=alpha, label=f"{c} (forget)",c="red", edgecolors="red", linewidths=0.5)

        
    #plt.title(title)
    plt.xticks([]); plt.yticks([])
    #plt.legend(loc="best", fontsize=25, ncol=2)
    plt.legend(loc="upper right",markerscale=5, fontsize=25)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        print(f"[save] {out_path}")
    else:
        plt.show()
    plt.close()


# -------------------------- expressextractget --------------------------
@torch.no_grad()
def get_features_from_head_input(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    normalize: bool = True,
    pool_mode: str = "avg",
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """
    useuse collect_features_for_head “classification headoutputinput”'s/offeature（t-SNE use）。
    Return：X[N, d], y[N], used(str), d_used(int)
    """
    feats, labels, used, d_used = collect_features_for_head(
        model, loader, device, pool_mode=pool_mode, normalize=normalize
    )
    print(f"[features] source={used}, shape={tuple(feats.shape)}")
    return feats.cpu().numpy(), labels.cpu().numpy(), used, d_used


@torch.no_grad()
def get_classifier_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    optional：classificationdeviceoutput（logits）docompare t-SNE。
    if model(x) Return dict/tuple，byyour forward adapt。
    """
    model.eval().to(device, non_blocking=True)
    outs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            out = out.get("logits", None)
            if out is None:
                raise RuntimeError("model(x) Returncharacterfindto 'logits' ；byyour forward doadapt。")
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"methodparse forward outputclasstype：{type(out)}；byyourmodeldoadapt。")
        outs.append(out.detach().float().cpu())
        ys.append(yb.detach().cpu())
    logits = torch.cat(outs, 0).numpy()
    labels = torch.cat(ys, 0).numpy()
    print(f"[logits] shape={logits.shape}")
    return logits, labels


# -------------------------- layer --------------------------
def _resolve_save_paths(
    dataset: str,
    unlearn_method: str,
    unlearn_classes: Sequence[int],
    out_root: str,
) -> Tuple[str, str]:
    cls_part = ",".join(map(str, unlearn_classes)) if len(unlearn_classes) > 0 else "none"
    method_dir = os.path.join(out_root, unlearn_method)
    base = f"{dataset}_{cls_part}"
    out_feat = os.path.join(method_dir, f"{base}.png")
    out_logit = os.path.join(method_dir, f"{base}-logits.png")
    return out_feat, out_logit


def tsne_visualize(
    args,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    out_root: str = "result/plot_T_SNE",
    perplexity: float = 30.0,
    max_points: int = 10000,
    pca_dim: int = 50,
    n_iter: int = 1000,
    seed: int = 0,
    also_logits: bool = True,
) -> None:
    """
    complete in one call：extract“feature vector”（classification headoutputinput）→ t-SNE → maintain；
    optionalagaincompare logits 's/of t-SNE。
    need args at leastcontain：args.dataset, args.unlearn_method, args.unlearn_class（list or commastring）
    """
    seed_everything(seed)

    # parseclasscolumnexpress
    if hasattr(args, "unlearn_class") and isinstance(args.unlearn_class, str):
        unlearn_classes = [int(x) for x in args.unlearn_class.split(",")] if args.unlearn_class else []
    else:
        unlearn_classes = list(getattr(args, "unlearn_class", []) or [])

    # outputpath
    out_feat, out_logit = _resolve_save_paths(
        dataset=getattr(args, "dataset", "unknown"),
        unlearn_method=getattr(args, "unlearn_method", "unknown"),
        unlearn_classes=unlearn_classes,
        out_root=out_root,
    )
    
    # 1) Features（classification headoutputinput）
    X_feat, y, used, d_used = get_features_from_head_input(
        model, loader, device, normalize=True, pool_mode="avg"
    )

    '''''''''
    need_cmf_prep = (
        "CMF" in str(getattr(args, "unlearn_method", "")) and
        (used.startswith("encoder") or used.startswith("extract_features")) and
        hasattr(model, "_preprocess_feats_for_cmf")
    )
    
    if need_cmf_prep:
        print("[t-SNE] tested CMF preprocessing on features ...")
        with torch.no_grad():
            model.eval().to(device, non_blocking=True)
            X_buf, y_buf = [], []
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                # all“encoder originalfeature”
                if hasattr(model, "extract_features"):
                    f = model.extract_features(xb)        # [B, ?]
                else:
                    # one：directlybeforeinbetweenfeatureline/execute，onehave extract_features
                    f = xb
                z = model._preprocess_feats_for_cmf(f)    # thisindo normalize/inify/again normalize
                # surekeepistwodimension [B, D]
                if z.dim() > 2:
                    z = z.flatten(1)
                X_buf.append(z.detach().cpu())
                y_buf.append(yb.detach().cpu())
            X_cmf = torch.cat(X_buf, 0).float().numpy()
            y_cmf = torch.cat(y_buf, 0).long().numpy()

        # use CMF logicafter's/offeaturereplace
        X_feat, y = X_cmf, y_cmf

    '''''''''
    X_feat, y_feat = _subsample(X_feat, y, max_points=max_points, seed=seed)
    Z_feat = compute_tsne(
        X_feat, use_pca=True, pca_dim=pca_dim,
        perplexity=perplexity, n_iter=n_iter, seed=seed
    )

    # save npz
    npz_path = out_feat.replace(".png", ".npz")
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, Z=Z_feat, y=y_feat)
    print(f"[t-SNE] saved embedding to {npz_path}")
    
    plot_scatter_2d(Z_feat, y_feat, f"t-SNE on Features ({getattr(args, 'dataset', '')})", out_feat, forget_classes=args.unlearn_class)

    '''''
    # 2) (optional)Logits compare
    if also_logits:
        try:
            X_logit, y2 = get_classifier_logits(model, loader, device)
            X_logit, y_logit = _subsample(X_logit, y2, max_points=max_points, seed=seed)
            Z_logit = compute_tsne(
                X_logit, use_pca=False,  # logits dimensiondegreepass=classnumber，generally no need PCA
                perplexity=perplexity, n_iter=n_iter, seed=seed
            )
            plot_scatter_2d(Z_logit, y_logit, f"t-SNE on Logits ({getattr(args, 'dataset', '')})", out_logit)
        except Exception as e:
            print(f"[t-SNE] pass logits ：{e}")
    '''''
