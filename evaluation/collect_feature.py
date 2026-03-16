# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Iterable, List, Dict, Any
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def _root_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def find_head_module(model: nn.Module) -> Tuple[str, nn.Module]:
    m = _root_model(model)
    for name in ["CMFweights", "classifier", "fc", "head"]:
        if hasattr(m, name):
            mod = getattr(m, name)
            if isinstance(mod, nn.Linear):
                return name, mod
            W = getattr(mod, "weight", None)
            if isinstance(W, torch.Tensor) and W.dim() == 2:
                return name, mod
            for subn, subm in mod.named_modules():
                if isinstance(subm, nn.Linear):
                    return f"{name}.{subn}", subm
                W = getattr(subm, "weight", None)
                if isinstance(W, torch.Tensor) and W.dim() == 2:
                    return f"{name}.{subn}", subm
    # fallback：at rootfindnon- encoder 's/of Linear/2D weight
    encoder_prefixes = []
    if hasattr(m, "encoder") and isinstance(m.encoder, nn.Module):
        for n, _ in m.encoder.named_modules():
            encoder_prefixes.append("encoder" if n == "" else f"encoder.{n}")
    candidates: List[Tuple[str, nn.Module]] = []
    for n, mod in m.named_modules():
        W = getattr(mod, "weight", None)
        ok = isinstance(mod, nn.Linear) or (isinstance(W, torch.Tensor) and W.dim() == 2)
        if ok and not any(n == p or n.startswith(p + ".") for p in encoder_prefixes):
            candidates.append((n, mod))
    if not candidates:
        raise RuntimeError("No linear/2D-weight head found (e.g., CMFweights/classifier/fc/head).")
    return candidates[-1]


@torch.no_grad()
def collect_features_for_head(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    pool_mode: str = "avg",   # only whenneed to 4D poolingto 2D timeuseuse：'avg' | 'max' | 'flatten'
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
    """
    Try first：in head hook on“outputinput”（approachA）。
     head notcalluse（ CMFweights onlymaintainweightnotparticipate forward）or batch nottofeature，
    selffallback：directlyuse model.extract_features(x) orin encoder hook output on，andby head column numberdopooling/flattenmatch。
    Return:
      feats  [N, d_used]  （and head.weight.shape[1] consistent）
      labels [N]
      used   getcome
      d_used actually/realfeaturedimensiondegree
    """
    m = _root_model(model)
    model.eval().to(device, non_blocking=True)

    # 1) get head othercolumn number d_head
    head_name, head_mod = find_head_module(model)
    W = getattr(head_mod, "weight", None)
    if not (isinstance(W, torch.Tensor) and W.dim() == 2):
        raise RuntimeError(f"Head '{head_name}' weightnotis 2D 。")
    K, d_head = W.shape

    feats_buf: List[torch.Tensor] = []
    labels_buf: List[torch.Tensor] = []
    used = None

    # ---------- work：putmeaning out form [B, d_head] ----------
    def _to_dhead(out: torch.Tensor) -> torch.Tensor:
        if out.dim() == 2:
            f = out
        elif out.dim() == 4:
            B, C, H, W_ = out.shape
            if d_head == C:
                f = F.adaptive_avg_pool2d(out, 1).flatten(1)      # [B, C]
            elif d_head == C * H * W_:
                f = out.flatten(1)                                # [B, C*H*W]
            else:
                if pool_mode == "avg":
                    f = F.adaptive_avg_pool2d(out, 1).flatten(1)
                elif pool_mode == "max":
                    f = F.adaptive_max_pool2d(out, 1).flatten(1)
                elif pool_mode == "flatten":
                    f = out.flatten(1)
                else:
                    raise ValueError(f"Unknown pool_mode: {pool_mode}")
        else:
            f = out.flatten(1)
        return f

    # ---------- approachA：in head hook on“outputinput” ----------
    def _collect_via_head_input(one_pass_only=False) -> bool:
        nonlocal used
        feats_tmp: List[torch.Tensor] = []
        def _hook_head_input(mod, inp, out):
            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            if isinstance(x, torch.Tensor):
                feats_tmp.append(_to_dhead(x).detach().cpu())

        handle = head_mod.register_forward_hook(_hook_head_input)
        used = f"head::<input>::{head_name}"

        got_any = False
        first = True
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)
            labels_buf.append(yb.detach().cpu())
            if first:
                first = False
                if len(feats_tmp) > 0:
                    got_any = True
                    if one_pass_only:
                        break
                else:
                    #  batch to， head notcalluse
                    break
        handle.remove()

        if got_any:
            feats_buf.extend(feats_tmp)
            return True
        else:
            # append's/of labels（is/betofeature）
            if len(labels_buf) > 0:
                labels_buf.clear()
            return False

    # ---------- fallback1：directlycalluse extract_features ----------
    def _collect_via_extract_features() -> bool:
        nonlocal used
        if not hasattr(model, "extract_features"):
            return False
        used = "extract_features::<call>"
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            feats = model.extract_features(xb)            # [B, ?]
            feats = _to_dhead(feats)                      # → [B, d_head]
            feats_buf.append(feats.detach().cpu())
            labels_buf.append(yb.detach().cpu())
        return len(feats_buf) > 0

    # ---------- fallback2：in encoder hook on“output” ----------
    def _collect_via_encoder_output() -> bool:
        nonlocal used
        if not hasattr(m, "encoder"):
            return False
        feats_tmp: List[torch.Tensor] = []
        def _hook_enc_out(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                feats_tmp.append(_to_dhead(out).detach().cpu())

        handle = m.encoder.register_forward_hook(_hook_enc_out)
        used = "encoder::<output>"

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            _ = model(xb)
            labels_buf.append(yb.detach().cpu())
        handle.remove()

        if len(feats_tmp) == 0:
            return False
        feats_buf.extend(feats_tmp)
        return True

    # ========== line/execute：firstA；notline/executeagainfallback ==========
    ok = _collect_via_head_input(one_pass_only=False)
    if not ok:
        # is CMFweights calluse；tryfallbackapproach
        ok = _collect_via_extract_features() or _collect_via_encoder_output()

    if not ok or len(feats_buf) == 0:
        raise RuntimeError(f"[collect] Hook/fallbackaveragetofeature，used={used}。"
                           f"useuse CMFweights，fallback extract_features/encoder output。")

    feats = torch.cat(feats_buf, dim=0).float()
    labels = torch.cat(labels_buf, dim=0).long()
    d_used = int(feats.size(1))

    if normalize and feats.numel() > 0:
        feats = F.normalize(feats, dim=1)

    # （optionalverify）
    # assert d_used == d_head, f"featuredimensiondegree {d_used} and head.weight column number {d_head} notconsistent"



    print(f"[collect] got features [N={feats.size(0)}, d_used={d_used}] for head '{head_name}' [K={K}, d_head={d_head}], used={used}")

    return feats, labels, used, d_used