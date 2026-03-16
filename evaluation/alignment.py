# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Iterable, List, Dict, Any
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from evaluation.collect_feature import collect_features_for_head, find_head_module

# ===================== Utility =====================
def _root_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

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

    dotted = dotted.lstrip('.')
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

def _pool_4d(x: torch.Tensor, mode: str = "avg") -> torch.Tensor:
    if mode == 'avg':
        return F.adaptive_avg_pool2d(x, (1,1)).flatten(1)  # [B,C,H,W]->[B,C]
    elif mode == 'flatten':
        return x.flatten(1)                                # [B, C*H*W]
    else:
        raise ValueError(f"Unknown pool mode: {mode}")

def fro_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x, ord="fro")

def etf_matrix(K: int, device: torch.device) -> torch.Tensor:
    if K < 2:
        raise ValueError("ETF undefined for K < 2.")
    I = torch.eye(K, device=device)
    one = torch.ones((K, K), device=device)
    return (I - one / K) / float(np.sqrt(K - 1))


# ===================== ：hook fixedlayer“output”extractfeature =====================
@torch.no_grad()
def _extract_layer_features(model, loader, layer_name='flatten', device='cpu', pool='avg'):
    """
    infixedlayer forward hook，extractlayeroutputanddopooling/flatten：
      - pool='avg'     → 4D do GAP → [B,C]
      - pool='flatten' → 4D flatten → [B, C*H*W]
      - 2D directlyuseuse
    Return：X [N,D] float32 (CPU), y [N] long (CPU)
    """
    target, used_name = _resolve_module(model, layer_name, return_name=True)
    feats, labels = [], []

    print(target, used_name)

    def _hook(m, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        if out.dim() == 4:
            f = _pool_4d(out, pool)
        elif out.dim() == 2:
            f = out
        else:
            f = out.flatten(1)
        feats.append(f.detach().cpu())

    handle = target.register_forward_hook(_hook)
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        labels.append(y.detach().cpu())
    handle.remove()

    if len(feats) == 0:
        raise RuntimeError(f"[extract] empty hook at layer={used_name}")

    X = torch.cat(feats, dim=0).float()
    y = torch.cat(labels, dim=0).long()
    return X, y

def _find_prev_module(model: nn.Module, target_name: str) -> Tuple[Optional[str], Optional[nn.Module]]:
    """
    in DFS 's/of named_modules() in，Return target_name 's/ofbeforeone (name, module)。
    if target_name notincolumnorbeforeonenotmaintainin，Return (None, None)。
    """
    prev = (None, None)
    for name, mod in model.named_modules():
        print(name)
        if name == target_name:
            return prev
        prev = (name, mod)
    return (None, None)



# ===================== classaveragevalue/global mean =====================
def compute_class_means(
    feats: torch.Tensor,
    labels: torch.Tensor,
    K: Optional[int] = None,
    *,
    samplewise_normalized: bool,
    center_H: bool = True,
    row_normalize_output: bool = True,
) -> Dict[str, torch.Tensor]:
    if feats.ndim != 2:
        raise ValueError(f"feats should be [N,d], got {feats.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels should be [N], got {labels.shape}")
    if K is None:
        K = int(labels.max().item() + 1) if labels.numel() > 0 else 0

    N, d = feats.shape
    H_raw = torch.zeros(K, d, dtype=feats.dtype)
    counts = torch.zeros(K, dtype=torch.long)

    for c in range(K):
        idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            H_raw[c].fill_(float('nan'))
            counts[c] = 0
        else:
            H_raw[c] = feats.index_select(0, idx).mean(dim=0)
            counts[c] = idx.numel()

    mu_H = feats.mean(dim=0) if feats.size(0) > 0 else torch.zeros(d, dtype=feats.dtype)
    H_centered = (H_raw - mu_H.unsqueeze(0)) if center_H else None

    if row_normalize_output:
        mask_raw = ~torch.any(torch.isnan(H_raw), dim=1)
        H_raw[mask_raw] = F.normalize(H_raw[mask_raw], dim=1)
        if H_centered is not None:
            mask_ctr = ~torch.any(torch.isnan(H_centered), dim=1)
            H_centered[mask_ctr] = F.normalize(H_centered[mask_ctr], dim=1)

    return {"H_raw": H_raw, "H_centered": H_centered, "mu_H": mu_H, "counts": counts}

# ===================== singleclassto/foralign（classaveragevalueglobal mean） =====================
@torch.no_grad()
def per_class_alignment(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    pool_mode: str = "avg",
    layer_name: Optional[str] = None,
    save_debug: Optional[str] = None, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    s_c = < ŵ_c , normalize(h_c - μ) >
    - W：comeself head（optimalfirst CMFweights）and L2 line/executenormoneify
    - feature：comeself head before's/ofonelayeroutput（self/fixed），firstsamplethis L2 normoneify
    - h_c：classaveragevalue；μ：samplethisaveragevalue；to/for (h_c-μ) line/executenormoneifyafterand ŵ_c dot
    Return：s_diag[K], counts[K]
    """
    head_name, head_mod = find_head_module(model)
    W = getattr(head_mod, "weight", None)
    if not (isinstance(W, torch.Tensor) and W.dim() == 2):
        raise RuntimeError(f"Head '{head_name}' has no 2D weight.")
    W = W.detach()
    K, d = W.shape
    W_hat = F.normalize(W, dim=1)

    feats, labels, used, d_used = collect_features_for_head(
        model, loader, device,
        pool_mode=pool_mode, normalize=True
    )
    if d_used != d:
        raise RuntimeError(f"Feature dim {d_used} != head dim {d} (used layer: {used})")

    cm = compute_class_means(
        feats=feats, labels=labels, K=K,
        samplewise_normalized=True, center_H=True, row_normalize_output=True
    )
    Hc = cm["H_centered"]; counts = cm["counts"]
    s_diag = torch.full((K,), float('nan'), dtype=torch.float32)
    valid = counts > 0
    if valid.any():
        s_diag[valid] = (W_hat[valid] * Hc[valid].to(W_hat.device)).sum(dim=1).float().cpu()

    if save_debug:
        pack = dict(
            used_source=used,
            head_name=head_name,
            feats=feats.cpu(),
            labels=labels.cpu(),
            H_centered=Hc.cpu(),
            H_raw=cm["H_raw"].cpu(),
            mu_H=cm["mu_H"].cpu(),
            counts=counts.cpu(),
            W=W.cpu(),
            W_hat=W_hat.cpu(),
            note="new_per_class_alignment",
        )
        torch.save(pack, save_debug)
        print(f"[per_class_alignment] debug saved to: {save_debug}  (used={used})")


    return s_diag.cpu(), counts.cpu()

# ===================== separategroup NC3 =====================
@torch.no_grad()
def grouped_nc3(
    model: nn.Module,
    loader,
    device: torch.device,
    retain_classes: Iterable[int],
    forget_classes: Iterable[int],
    *,
    center_mode: str = "subset",   # "subset"|"global"
    pool_mode: str = "avg",
    normalize_feats: bool = True,
    layer_name: Optional[str] = None,
) -> Dict[str, Any]:
    head_name, head_mod = find_head_module(model)
    W_full = getattr(head_mod, "weight", None)
    if not (isinstance(W_full, torch.Tensor) and W_full.dim() == 2):
        raise RuntimeError(f"Head '{head_name}' has no 2D weight.")
    W_full = W_full.detach().to(device)
    K_total, d = W_full.shape

    feats, labels, used, d_used = collect_features_for_head(
        model, loader, device,
        pool_mode=pool_mode, normalize=normalize_feats
    )
    if d_used != d:
        raise RuntimeError(f"Feature dim {d_used} != head dim {d} (used layer: {used})")
    feats = feats.to(device); labels = labels.to(device)

    C_f = sorted(set(int(c) for c in forget_classes))
    C_all = list(range(K_total))
    C_r = sorted([c for c in C_all if c not in C_f])
    K_r, K_f = len(C_r), len(C_f)

    cm_all = compute_class_means(
        feats=feats.cpu(), labels=labels.cpu(), K=K_total,
        samplewise_normalized=normalize_feats, center_H=False, row_normalize_output=False
    )
    H_raw_all = cm_all["H_raw"]

    def _take(mat: torch.Tensor, rows: List[int]) -> Optional[torch.Tensor]:
        return mat[rows].contiguous() if len(rows) > 0 else None

    def _center_and_norm(H_raw_subset: Optional[torch.Tensor], mode: str) -> Optional[torch.Tensor]:
        if H_raw_subset is None:
            return None
        H = H_raw_subset.clone()
        valid = ~torch.any(torch.isnan(H), dim=1)
        if not valid.any():
            return H
        if mode == "subset":
            mu = H[valid].mean(dim=0, keepdim=True)
        elif mode == "global":
            mu = feats.mean(dim=0, keepdim=True).cpu()
        else:
            raise ValueError(f"Unknown center_mode: {mode}")
        H[valid] = F.normalize(H[valid] - mu, dim=1)
        return H

    H_r = _center_and_norm(_take(H_raw_all, C_r), center_mode) if K_r > 0 else None
    H_f = _center_and_norm(_take(H_raw_all, C_f), center_mode) if K_f > 0 else None
    W_r = W_full[C_r, :] if K_r > 0 else None
    W_f = W_full[C_f, :] if K_f > 0 else None

    def _nc3(Ws: Optional[torch.Tensor], Hs: Optional[torch.Tensor]):
        if Ws is None or Hs is None:
            return None, None
        K_s = Ws.size(0)
        if K_s < 1:
            return None, None
        A = Ws @ Hs.to(Ws.device).T
        A = A / (fro_norm(A) + 1e-12)
        if K_s == 1:
            return A, None
        ETF = etf_matrix(K_s, Ws.device)
        return A, float(fro_norm(A - ETF).item())

    A_r, NC3_r = _nc3(W_r, H_r)
    A_f, NC3_f = _nc3(W_f, H_f)

    num = 0.0; den = 0.0
    if NC3_r is not None: num += K_r * NC3_r; den += K_r
    if NC3_f is not None: num += K_f * NC3_f; den += K_f
    NC3_group = float(num/den) if den > 0 else None

    # get retain class C_r（you grouped_nc3 inhave）
    W = W_full[C_r].detach().cpu()       # [K_r, d]
    H = H_r.detach().cpu()               # [K_r, d]  line/executesinglepositionify（byyouractually/real）

    # 1) weightline/executebetweendegree（is ~1）
    print(W.shape, H.shape)
    W_cos = F.normalize(W, dim=1) @ F.normalize(W, dim=1).T
    print("W_cos.shape", W_cos.shape)
    print("mean offdiag cos(W_r rows):", W_cos)

    # 2) H line/executebetween（is ~1）
    H_cos = H @ H.T
    print("H_cos.shape", H_cos.shape)
    print("mean offdiag cos(H_r rows):", H_cos)

    # 3) W 's/of
    print("rank(W_r):", torch.linalg.matrix_rank(W).item())

    # 4) A do Fro normoneifybeforeis-1
    A0 = (W @ H.T).cpu()
    print("rank(W_r @ H_r^T):", torch.linalg.matrix_rank(A0).item())

    # 5) W line/executenumberseparate
    print("||W_r|| (min/mean/max):",
        W.norm(dim=1).min().item(), W.norm(dim=1).mean().item(), W.norm(dim=1).max().item())


    return dict(
        head=head_name, used_feature_from=used,
        feature_dim=d, K_total=K_total, K_r=K_r, K_f=K_f,
        center_mode=center_mode,
        A_r=A_r, A_f=A_f,
        NC3_r=NC3_r, NC3_f=NC3_f, NC3_group=NC3_group,
    )
