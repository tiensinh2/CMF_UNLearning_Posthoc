# evaluation/nc_cmf.py
# Unified NC metric：
#   - NC-4: NCC accuracy (nearest class center)
#   - NC-3: feature-classifier misalignment (duality distance style)
#
# Support two types of feature spaces：
#   1) CMF： z = _preprocess_feats_for_cmf(extract_features(x))
#   2) non-CMF：head input features（and classifier.weight dimensiondegreeconsistent）

from typing import Dict, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Depends on your existing linear probe tool
from evaluation.linear_prob import (
    _extract_cmf_geometry_features,
    _extract_head_input_features,
)

# ----------------- Utility -----------------

@torch.no_grad()
def _features_for_nc(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    use_cmf: bool,
    pool_mode: str = "avg",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified feature extraction interface：
      - use_cmf=True:  Use CMF geometry (ẑ)， _extract_cmf_geometry_features
      - use_cmf=False: useuse head input features (and classifier.weight dimensiondegreeconsistent)
    Return:
      - X: [N, D]  float32
      - y: [N]     long
    """
    model.eval().to(device)

    if use_cmf:
        X, y = _extract_cmf_geometry_features(model, loader, device)
    else:
        # Do not normalize, keep original geometry consistent with classifier
        X, y, used, d_used = _extract_head_input_features(
            model=model,
            loader=loader,
            device=device,
            pool_mode=pool_mode,
            normalize=False,
        )
        # X, y already CPU tensor，thisinkeep interface consistent
    return X.float(), y.long()


def _compute_mu_G_and_mu_c(
    X: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Given feature X[N,D] andlabel y[N]，compute：
      - global mean μ_G
      - per-class mean dictionary mu_c_dict[c] = μ_c
    """
    mu_G = X.mean(dim=0)  # [D]
    mu_c_dict: Dict[int, torch.Tensor] = {}
    classes = torch.unique(y).tolist()
    for c in classes:
        c_int = int(c)
        mask = (y == c_int)
        if mask.any():
            mu_c_dict[c_int] = X[mask].mean(dim=0)
    return mu_G, mu_c_dict


def get_classifier_weights(
    model: nn.Module,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returnclassificationdeviceweightmatrix W[C,D]：
      - Try first CMF cmf_weights in model / CMFweights.weight
      - then try model.classifier / model.fc / model.head / model.linear
      - then try againlast nn.Linear in module
      - final fallback：inhavenumber / buffer insearch 2D tensors，one dimension equals num_classes
    """
    # 1) If model itself exposes cmf_weights attribute（for exampleyou add it in the future）
    if hasattr(model, "cmf_weights"):
        W = getattr(model, "cmf_weights").detach().to(device)
        if W.ndim == 2:
            if W.shape[0] == num_classes:
                return W
            if W.shape[1] == num_classes:
                return W.t()

    # 2) Your actual CMF model case：has model.CMFweights.weight this buffer
    if hasattr(model, "CMFweights"):
        cmf_mod = getattr(model, "CMFweights")
        if hasattr(cmf_mod, "weight"):
            W = cmf_mod.weight.detach().to(device)
            if W.ndim == 2:
                if W.shape[0] == num_classes:
                    return W
                if W.shape[1] == num_classes:
                    return W.t()

    # 3) common head names：classifier / fc / head / linear
    for attr in ["classifier", "fc", "head", "linear"]:
        if hasattr(model, attr):
            head = getattr(model, attr)
            W = None
            if isinstance(head, nn.Linear):
                W = head.weight
            elif hasattr(head, "weight") and isinstance(head.weight, torch.Tensor):
                # Some custom heads are not Linear，but have 2D weight
                W = head.weight
            if W is not None and W.ndim == 2:
                W = W.detach().to(device)
                if W.shape[0] == num_classes:
                    return W
                if W.shape[1] == num_classes:
                    return W.t()

    # 4) fallback: last nn.Linear in module（for common ResNet / Linear head）
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        W = last_linear.weight.detach().to(device)
        if W.ndim == 2:
            if W.shape[0] == num_classes:
                return W
            if W.shape[1] == num_classes:
                return W.t()

    # 5) afterkeep：inhavenumberinfind 2D weight，one dimension equals num_classes
    candidate = None
    for name, p in model.named_parameters():
        if p.ndim == 2 and (p.shape[0] == num_classes or p.shape[1] == num_classes):
            candidate = p
            # haveneedcan：
            # print(f"[NC] Using parameter '{name}' as classifier weight.")
            break

    # ifnumberinfindnotto，again buffer infind（for example CMFweights.weight this register_buffer）
    if candidate is None:
        for name, b in model.named_buffers():
            if b.ndim == 2 and (b.shape[0] == num_classes or b.shape[1] == num_classes):
                candidate = b
                # print(f"[NC] Using buffer '{name}' as classifier weight.")
                break

    if candidate is None:
        raise RuntimeError(
            "No Linear classifier head or suitable 2D weight found in model."
        )

    W = candidate.detach().to(device)
    if W.shape[0] == num_classes:
        return W
    if W.shape[1] == num_classes:
        return W.t()
    raise RuntimeError(
        f"Found 2D weight {W.shape} but neither dim equals num_classes={num_classes}"
    )


# ----------------- NC-3: duality distance -----------------

def duality_distance(
    mu_c_dict: Dict[int, torch.Tensor],
    mu_G: torch.Tensor,
    W: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    use nc3_measurement.py in's/of duality distance ：
      - to/forclass c: h_c = μ_c - μ_G
      - line/executenormoneify h and W
      - compute (1 - cos(h_c, w_c)) 's/ofclassaverage

    Return:
      - h_mean_norm: ‖h_mean‖_2
      - w_norm     : ‖W‖_2
      - distance   : average (1 - cos)， misaligned
    """
    if len(mu_c_dict) == 0:
        return 0.0, float(torch.norm(W).item()), 0.0

    C, D = W.shape
    device = W.device

    h_mean = torch.zeros_like(W, device=device)
    rows: Sequence[int] = []
    for c, mu_c in mu_c_dict.items():
        if 0 <= c < C:
            h_mean[c, :] = mu_c.to(device) - mu_G.to(device)
            rows.append(c)

    if len(rows) == 0:
        return float(torch.norm(h_mean).item()), float(torch.norm(W).item()), 0.0

    h_mean_norm = torch.norm(h_mean).item()
    w_norm = torch.norm(W).item()

    h_normed = F.normalize(h_mean, dim=1, p=2)
    W_normed = F.normalize(W, dim=1, p=2)

    dist = 0.0
    for i in rows:
        dist += 1.0 - torch.dot(h_normed[i, :], W_normed[i, :]).item()
    dist /= float(len(rows))

    return float(h_mean_norm), float(w_norm), float(dist)


# ----------------- NC-4: NCC accuracy -----------------

@torch.no_grad()
def ncc_accuracy_from_features(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    num_classes: int,
) -> Tuple[float, torch.Tensor]:
    """
    feature NCC：
      - use X_tr, y_tr computeclassaveragevalue μ_c
      - X_eval classify using nearest class center
      - Returnpreparesure + classpreparesure
    """
    device = X_tr.device
    C = num_classes

    means = []
    for c in range(C):
        mask = (y_tr == c)
        if mask.any():
            mu_c = X_tr[mask].mean(dim=0)
        else:
            mu_c = torch.zeros(X_tr.size(1), device=device)
        means.append(mu_c)
    M = torch.stack(means, dim=0)        # [C,D]
    M_exp = M.unsqueeze(0)               # [1,C,D]

    diffs = X_eval.unsqueeze(1) - M_exp  # [N,C,D]
    dists = torch.norm(diffs, dim=2)     # [N,C]
    pred = dists.argmin(dim=1)           # [N]

    correct = (pred == y_eval).float()
    acc_all = correct.mean().item()

    acc_per_class = torch.zeros(C, dtype=torch.float, device=device)
    for c in range(C):
        m = (y_eval == c)
        if m.any():
            acc_per_class[c] = correct[m].mean()
        else:
            acc_per_class[c] = float("nan")

    return float(acc_all), acc_per_class.cpu()


@torch.no_grad()
def ncc_mismatch(
    args,
    model: nn.Module,
    train_loader,
    eval_loader,
    device: torch.device,
    *,
    pool_mode: str = "avg",
) -> Dict[str, object]:
    """
    useuse train_loader is/be“classin”'s/of，eval_loader is/beevaluate，
    computeclassinclassificationdevice (NCC) in eval_loader 's/ofexpress。

    thisinputyouoriginalcome ncc_mismatch_cmf.py in's/of
        loader1 = train_loader
        loader2 = forget/retain_test_loader
    's/of NCC ，tounified/statisticone's/of NC/CMF income。:contentReference[oaicite:4]{index=4}

    Return:
      - ncc_acc           : NCC accuracy (0~1)
      - ncc_mismatch      : 1 - accuracy
      - ncc_acc_per_class : [C] classpreparesure (convenientafter)
    """
    device = device if isinstance(device, torch.device) else torch.device(device)

    # fixedisin CMF getfeature（and compute_nc_all keepsupportconsistent）
    use_cmf = ("CMF" in getattr(args, "unlearn_method", "")) and hasattr(
        model, "_preprocess_feats_for_cmf"
    )

    # 1) useunified/statisticonegetfeature
    X_tr, y_tr = _features_for_nc(
        model=model,
        loader=train_loader,
        device=device,
        use_cmf=use_cmf,
        pool_mode=pool_mode,
    )
    X_ev, y_ev = _features_for_nc(
        model=model,
        loader=eval_loader,
        device=device,
        use_cmf=use_cmf,
        pool_mode=pool_mode,
    )

    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    X_ev = X_ev.to(device)
    y_ev = y_ev.to(device)

    # 2) usehave's/of“feature NCC”comepreparesure + per-class
    acc_all, acc_per_class = ncc_accuracy_from_features(
        X_tr, y_tr, X_ev, y_ev, num_classes=args.num_classes
    )
    mismatch = 1.0 - acc_all

    return {
        "ncc_acc": float(acc_all),
        "ncc_mismatch": float(mismatch),
        "ncc_acc_per_class": acc_per_class.tolist(),
    }


@torch.no_grad()
def compute_nc_on_loader(
    args,
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    pool_mode: str = "avg",
) -> Dict[str, object]:
    """
    ingivefixed's/of loader compute：
      - NC-3: duality_distance(μ_c, μ_G, W)
      - NCC: in loader 's/ofclassinclassificationpreparesure

    usemethod：
      - canto/for train_loader / test_loader / test_retain_loader / test_forget_loader
        separatecalluse，toto/forapply split 's/of NC 。
    """
    device = device if isinstance(device, torch.device) else torch.device(device)

    # fixedisUse CMF geometry（and compute_nc_all keepsupportconsistent）
    use_cmf = ("CMF" in getattr(args, "unlearn_method", "")) and hasattr(
        model, "_preprocess_feats_for_cmf"
    )

    # classificationdeviceweight W[C,D]
    W = get_classifier_weights(model, args.num_classes, device=device)

    # feature + label
    X, y = _features_for_nc(
        model=model,
        loader=loader,
        device=device,
        use_cmf=use_cmf,
        pool_mode=pool_mode,
    )
    X = X.to(device)
    y = y.to(device)

    # NC-3: feature–classifier misalignment
    muG, mu_c = _compute_mu_G_and_mu_c(X, y)
    h_norm, w_norm, dist = duality_distance(mu_c, muG, W)

    # NCC (NC-4)
    ncc_all, ncc_per_class = ncc_accuracy_from_features(
        X, y, X, y, args.num_classes
    )

    return {
        "nc3": float(dist),
        "nc3_h_norm": float(h_norm),
        "nc3_w_norm": float(w_norm),
        "ncc": float(ncc_all),
        "ncc_per_class": ncc_per_class.tolist(),
    }

# ----------------- layer：in train/retain/forget unified/statisticone NC -----------------

@torch.no_grad()
def compute_nc_all(
    args,
    model: nn.Module,
    train_loader,
    retain_loader,
    forget_loader,
    device: torch.device,
    *,
    pool_mode: str = "avg",
) -> Dict[str, float]:
    """
    inbeforetrainingseparate：
      - NCC (NC-4)：train_all / train_retain / train_forget
      - NC-3：samplesubset's/of duality distance

    ：
      - to/for CMF model（unlearn_method  "CMF" have _preprocess_feats_for_cmf）：
          Use CMF geometryfeature ẑ
      - to/fornon-CMF model：
          useuse head input features（and classifier.weight dimensiondegreeconsistent）
    """
    device = device if isinstance(device, torch.device) else torch.device(device)

    # fixedisUse CMF geometry
    use_cmf = ("CMF" in getattr(args, "unlearn_method", "")) and hasattr(
        model, "_preprocess_feats_for_cmf"
    )

    # classificationdeviceweight W[C,D]
    W = get_classifier_weights(model, args.num_classes, device=device)

    results: Dict[str, float] = {}

    # ---- 1) train_all ----
    X_all, y_all = _features_for_nc(model, train_loader, device, use_cmf=use_cmf, pool_mode=pool_mode)
    X_all = X_all.to(device)
    y_all = y_all.to(device)

    muG_all, mu_c_all = _compute_mu_G_and_mu_c(X_all, y_all)
    h_norm, w_norm, dist = duality_distance(mu_c_all, muG_all, W)
    results["nc3_train_all"] = dist
    results["nc3_train_all_h_norm"] = h_norm
    results["nc3_train_all_w_norm"] = w_norm

    ncc_all, ncc_all_per_class = ncc_accuracy_from_features(
        X_all, y_all, X_all, y_all, args.num_classes
    )
    results["ncc_train_all"] = ncc_all
    #  per-class，canmaintainform list
    results["ncc_train_all_per_class"] = ncc_all_per_class.tolist()

    # ---- 2) train_retain ----
    if retain_loader is not None:
        X_ret, y_ret = _features_for_nc(model, retain_loader, device, use_cmf=use_cmf, pool_mode=pool_mode)
        X_ret = X_ret.to(device)
        y_ret = y_ret.to(device)

        muG_ret, mu_c_ret = _compute_mu_G_and_mu_c(X_ret, y_ret)
        h_norm_r, w_norm_r, dist_r = duality_distance(mu_c_ret, muG_ret, W)

        results["nc3_train_retain"] = dist_r
        results["nc3_train_retain_h_norm"] = h_norm_r
        results["nc3_train_retain_w_norm"] = w_norm_r

        ncc_ret, ncc_ret_per_class = ncc_accuracy_from_features(
            X_ret, y_ret, X_ret, y_ret, args.num_classes
        )
        results["ncc_train_retain"] = ncc_ret
        results["ncc_train_retain_per_class"] = ncc_ret_per_class.tolist()

    # ---- 3) train_forget ----
    if forget_loader is not None:
        X_f, y_f = _features_for_nc(model, forget_loader, device, use_cmf=use_cmf, pool_mode=pool_mode)
        X_f = X_f.to(device)
        y_f = y_f.to(device)

        muG_f, mu_c_f = _compute_mu_G_and_mu_c(X_f, y_f)
        h_norm_f, w_norm_f, dist_f = duality_distance(mu_c_f, muG_f, W)

        results["nc3_train_forget"] = dist_f
        results["nc3_train_forget_h_norm"] = h_norm_f
        results["nc3_train_forget_w_norm"] = w_norm_f

        ncc_f, ncc_f_per_class = ncc_accuracy_from_features(
            X_f, y_f, X_f, y_f, args.num_classes
        )
        results["ncc_train_forget"] = ncc_f
        results["ncc_train_forget_per_class"] = ncc_f_per_class.tolist()

    return results
