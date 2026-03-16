import torch
import torch.nn.functional as F
import scipy.linalg as scilin
import numpy as np
from torch.nn import Conv2d, Linear

def pool_tensor(tensor, mode='avg'):
    """
    If tensor is [B,C,H,W], immediately reduces it to [B,C] using a single
    C‐optimized call.  Otherwise returns it unchanged.
    """
    if tensor.dim() == 4:
        if mode == 'avg':
            # adaptive_avg_pool2d will give you a [B,C,1,1] tensor
            # which you can then squeeze to [B,C].
            out = F.adaptive_avg_pool2d(tensor, 1)
        else:
            out = F.adaptive_max_pool2d(tensor, 1)
        # remove those singleton spatial dims
        return out.view(out.size(0), -1)
    return tensor


def _class_means(feats, labels, classes):
    return torch.stack([feats[labels == c].mean(0) for c in classes])


def _build_Sw_Sb(feats, labels, classes, use_global):
    K = len(classes)
    mu_c = _class_means(feats, labels, classes)
    if use_global:
        mu_G = feats.mean(0)
    else:
        mask = torch.isin(labels, torch.tensor(classes, device=feats.device))
        mu_G = feats[mask].mean(0)
    Sw = torch.zeros((feats.size(1), feats.size(1)), device=feats.device)
    for i, c in enumerate(classes):
        xi = feats[labels == c]
        diff = xi - mu_c[i]
        Sw += (diff.t() @ diff) / xi.size(0)
    Sw /= K
    diffc = (mu_c - mu_G).T
    Sb = diffc @ diffc.T / K
    return Sw, Sb, mu_c, mu_G


def _nc1(Sw, Sb, K):
    return (torch.trace(Sw @ torch.tensor(scilin.pinv(Sb.cpu().numpy()), device=Sw.device)) / K).item()


def _nc2(mu_c, mu_G):
    K = mu_c.size(0)
    H = (mu_c - mu_G).T
    HH = H.T @ H
    HH /= torch.norm(HH, p='fro')
    ETF = (torch.eye(K, device=mu_c.device) - torch.ones(K, device=mu_c.device)/K) / ((K-1)**0.5)
    return torch.norm(HH - ETF, p='fro').item()


def _nc3_traditional(W, mu_c, mu_G):
    K = mu_c.size(0)
    H = (mu_c.to(W.device) - mu_G.to(W.device)).T
    WH = W @ H
    WHn = WH / torch.norm(WH, p='fro')
    ETF = (torch.eye(K, device=W.device) - torch.ones(K, device=W.device)/K) / ((K-1)**0.5)
    return torch.norm(WHn - ETF, p='fro').item()


def cdnv(feats, labels, classes):
    K = len(classes)
    if K < 2: return None
    mu = [feats[labels == c].mean(0) for c in classes]
    var = [(feats[labels == c] - mu[i]).pow(2).sum(1).mean() for i, c in enumerate(classes)]
    num, den = 0.0, 0.0
    for i in range(K):
        for j in range(i+1, K):
            d2 = (mu[i] - mu[j]).pow(2).sum()
            num += var[i] + var[j]
            den += 2 * d2
    return (num/den*2/(K*(K-1))).item()


def cos_stats(mu):
    if mu is None or mu.shape[0] < 2:
        return (None, None, None)
    c = (mu @ mu.T).cpu().numpy()
    off = c[np.triu_indices(c.shape[0], 1)]
    return (float(off.min()), float(off.max()), float(off.mean()))


def nc_metrics(model, loader, device, retain_classes, forget_classes):
    
    modules = [(name, m) for name, m in model.named_modules()
               if isinstance(m, (Conv2d, Linear))]
    if len(modules) < 2:
        raise RuntimeError("Model too small for NC metrics")

    # 2) Pick the last two as penultimate & final
    (pen_name, pen_mod), (fin_name, fin_mod) = modules[-2], modules[-1]

    pen_feats, fin_feats, labs = [], [], []
    def make_hook(buffer):
        def hook(m, inp, out):
            # use your pool_tensor helper for both 2D and 4D
            pooled = pool_tensor(out, mode='avg')   # 4D→[B,C], 2D→unchanged [B,C]
            buffer.append(pooled.detach().cpu())
        return hook
    
    h_pen = pen_mod.register_forward_hook(make_hook(pen_feats))
    h_fin = fin_mod.register_forward_hook(make_hook(fin_feats))

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            _ = model(x.to(device))
            labs.append(y)
    h_pen.remove()
    h_fin.remove()

    feats_pen = F.normalize(torch.cat(pen_feats, 0), dim=1)
    feats_fin = F.normalize(torch.cat(fin_feats, 0), dim=1)
    labels    = torch.cat(labs, 0)

    def build_sw_sb(f, l, classes, use_global):
        K = len(classes)
        mu_c = torch.stack([f[l==c].mean(0) for c in classes])
        mu_G = f.mean(0) if use_global else f[torch.isin(l, torch.tensor(classes, device=l.device))].mean(0)
        Sw = torch.zeros((f.size(1),f.size(1)), device=f.device)
        for i,c in enumerate(classes):
            xi = f[l==c]
            d = xi - mu_c[i]
            Sw += (d.t()@d)/xi.size(0)
        Sw /= K
        diff = (mu_c-mu_G).T
        Sb   = diff@diff.T / K
        return Sw, Sb, mu_c, mu_G

    def nc1(Sw, Sb, K):
        return (torch.trace(Sw @ torch.tensor(scilin.pinv(Sb.cpu().numpy()), device=Sw.device))/K).item()
    def nc2(mu_c, mu_G):
        K = mu_c.size(0)
        H = (mu_c-mu_G).T
        HH = H.T@H; HH/=torch.norm(HH,'fro')
        ETF = (torch.eye(K,device=mu_c.device)-torch.ones(K,device=mu_c.device)/K)/((K-1)**0.5)
        return torch.norm(HH-ETF,'fro').item()
    def nc3(W, mu_c, mu_G):
        K = mu_c.size(0)
        H = (mu_c-mu_G).T.to(W.device)
        WH = W@H; WH/=torch.norm(WH,'fro')
        ETF = (torch.eye(K,device=W.device)-torch.ones(K,device=W.device)/K)/((K-1)**0.5)
        return torch.norm(WH-ETF,'fro').item()
    def cdnv(f, l, classes):
        K=len(classes)
        if K<2: return None
        mu=[f[l==c].mean(0) for c in classes]
        var=[(f[l==c]-mu[i]).pow(2).sum(1).mean() for i,c in enumerate(classes)]
        num=den=0
        for i in range(K):
            for j in range(i+1,K):
                d2=(mu[i]-mu[j]).pow(2).sum()
                num+=var[i]+var[j]; den+=2*d2
        return (num/den*2/(K*(K-1))).item()
    def cos_stats(mu):
        if mu is None or mu.size(0)<2: return (None,None,None)
        c=(mu@mu.T).cpu().numpy()
        off=c[np.triu_indices(c.shape[0],1)]
        return float(off.min()), float(off.max()), float(off.mean())

    def compute_all(f, W):
        all_cls = sorted(set(retain_classes)|set(forget_classes))
        # NC1
        Sw_all, Sb_all, mu_all, muG_all = build_sw_sb(f, labels, all_cls, False)
        nc1_all    = nc1(Sw_all, Sb_all, len(all_cls))
        Sw_r, Sb_r, mu_r, muG_r = build_sw_sb(f, labels, retain_classes, True)
        nc1_ret    = nc1(Sw_r, Sb_r, len(retain_classes))
        nc1_for    = None
        if forget_classes:
            Sw_f, Sb_f, mu_f, muG_f = build_sw_sb(f, labels, forget_classes, True)
            nc1_for = nc1(Sw_f, Sb_f, len(forget_classes))
        # NC2
        nc2_all    = nc2(mu_all, muG_all)
        nc2_ret    = nc2(mu_r, muG_r)
        # NC3
        nc3_all    = nc3(W, mu_all, muG_all)
        # Delta
        WWT = (W@W.T).cpu().numpy()
        HH  = (W@(mu_all-muG_all).T.to(W.device)).cpu().numpy()
        Delta = np.abs(WWT-HH)
        def fro(idx):
            if not idx: return None
            sub = Delta[np.ix_(idx,idx)]
            return float(np.linalg.norm(sub,'fro'))
        idx_all = list(range(len(all_cls)))
        idx_r   = [all_cls.index(c) for c in retain_classes]
        idx_f   = [all_cls.index(c) for c in forget_classes]
        d_all   = fro(idx_all); d_ret=fro(idx_r); d_for=fro(idx_f)
        # cos‐stats
        cs_all = cos_stats((mu_all-muG_all))
        cs_ret = cos_stats((mu_all-muG_all)[idx_r]) if idx_r else (None,None,None)
        cs_for = cos_stats((mu_all-muG_all)[idx_f]) if idx_f else (None,None,None)
        # CDNV
        cd_all = cdnv(f, labels, all_cls)
        cd_ret = cdnv(f, labels, retain_classes)
        cd_for = cdnv(f, labels, forget_classes)

        return dict(
            NC1_all=nc1_all,   NC1_ret=nc1_ret,   NC1_for=nc1_for,
            NC2_all=nc2_all,   NC2_ret=nc2_ret,
            NC3_all=nc3_all,
            Delta_all=d_all,   Delta_ret=d_ret,   Delta_for=d_for,
            cos_all_min=cs_all[0], cos_all_max=cs_all[1], cos_all_avg=cs_all[2],
            cos_ret_min=cs_ret[0], cos_ret_max=cs_ret[1], cos_ret_avg=cs_ret[2],
            cos_for_min=cs_for[0], cos_for_max=cs_for[1], cos_for_avg=cs_for[2],
            CDNV_all=cd_all, CDNV_ret=cd_ret, CDNV_for=cd_for
        )
    
    W_final = fin_mod.weight.detach()
    metrics_pen = compute_all(feats_pen, W_final)
    #metrics_fin = compute_all(feats_fin, W_final)

    return metrics_pen








def nc_metrics_old(model, loader, device, retain_classes, forget_classes):

    if hasattr(model, 'avgpool'):
        h = model.avgpool.register_forward_hook(lambda m, i, o: pen.append(o.flatten(1).detach().cpu()))
    else:
        h = model.fc.register_forward_hook(lambda m, inp, o: pen.append(inp[0].detach().cpu()))


    pen, lab = [], []
    if hasattr(model, 'avgpool'):
        h = model.avgpool.register_forward_hook(lambda m, i, o: pen.append(o.flatten(1).detach().cpu()))
    else:
        h = model.fc.register_forward_hook(lambda m, inp, o: pen.append(inp[0].detach().cpu()))
    with torch.no_grad():
        for x, y in loader:
            _ = model(x.to(device))
            lab.append(y)
    h.remove()
    feats = F.normalize(torch.cat(pen, 0), dim=1)
    labels = torch.cat(lab, 0)
    all_cls = sorted(set(retain_classes) | set(forget_classes))
    Sw_all, Sb_all, mu_all, muG_all = _build_Sw_Sb(feats, labels, all_cls, False)
    NC1_all = _nc1(Sw_all, Sb_all, len(all_cls))
    Sw_r, Sb_r, mu_r, muG_r = _build_Sw_Sb(feats, labels, retain_classes, True)
    NC1_retain = _nc1(Sw_r, Sb_r, len(retain_classes))
    NC1_forget = None
    if len(forget_classes) > 0:
        Sw_f, Sb_f, mu_f, muG_f = _build_Sw_Sb(feats, labels, forget_classes, True)
        NC1_forget = _nc1(Sw_f, Sb_f, len(forget_classes))
    NC2_all = _nc2(mu_all, muG_all)
    NC2_retain = _nc2(mu_r, muG_r)
    W = model.fc.weight.detach()
    NC3 = _nc3_traditional(W, mu_all, muG_all)
    WWT = (W @ W.T).cpu().numpy()
    centered_mu_all = mu_all - muG_all
    HH_all = (W @ (centered_mu_all.to(W.device)).T).cpu().numpy()
    Delta = np.abs(WWT - HH_all)
    def fro(idx):
        if len(idx) == 0:
            return None
        sub = Delta[np.ix_(idx, idx)]
        return float(np.linalg.norm(sub, 'fro'))
    idx_all = list(range(len(all_cls)))
    idx_r = [all_cls.index(c) for c in retain_classes]
    idx_f = [all_cls.index(c) for c in forget_classes]
    Delta_all = fro(idx_all)
    Delta_retain = fro(idx_r)
    Delta_forget = fro(idx_f)
    cs_all = cos_stats(F.normalize(centered_mu_all, dim=1))
    if idx_r:
        centered_mu_ret = centered_mu_all[idx_r]
        cs_ret = cos_stats(F.normalize(centered_mu_ret, dim=1))
    else:
        cs_ret = (None, None, None)
    if idx_f:
        centered_mu_for = centered_mu_all[idx_f]
        cs_for = cos_stats(F.normalize(centered_mu_for, dim=1))
    else:
        cs_for = (None, None, None)
    CDNV_all = cdnv(feats, labels, all_cls)
    CDNV_retain = cdnv(feats, labels, retain_classes)
    CDNV_forget = cdnv(feats, labels, forget_classes)
    print(f"NC1_all={NC1_all:.4f}  NC1_retain={NC1_retain:.4f}  NC1_forget={NC1_forget:.4f}")
    print(f"NC2_all={NC2_all:.4f}  NC2_retain={NC2_retain:.4f}")
    print(f"NC3={NC3:.4f}")
    print(f"Delta_all={Delta_all:.4f}  Delta_retain={Delta_retain:.4f}  Delta_forget={Delta_forget:.4f}")
    print(f"cos_all    min={cs_all[0]:.4f} max={cs_all[1]:.4f} avg={cs_all[2]:.4f}")
    print(f"cos_retain min={cs_ret[0]:.4f} max={cs_ret[1]:.4f} avg={cs_ret[2]:.4f}")
    if cs_for[0] is not None:
        print(f"cos_forget min={cs_for[0]:.4f} max={cs_for[1]:.4f} avg={cs_for[2]:.4f}")
    else:
        print("cos_forget: N/A (need ≥2 forget classes)")
    # CDNV guard
    if CDNV_all    is not None: print(f"CDNV_all={CDNV_all:.4f}",    end='  ')
    else:                       print("CDNV_all: N/A",               end='  ')
    if CDNV_retain is not None: print(f"CDNV_retain={CDNV_retain:.4f}", end='  ')
    else:                       print("CDNV_retain: N/A",            end='  ')
    if CDNV_forget is not None: print(f"CDNV_forget={CDNV_forget:.4f}")
    else:                       print("CDNV_forget: N/A")
    metrics = dict(
        NC1_all=NC1_all, NC1_retain=NC1_retain, NC1_forget=NC1_forget,
        NC2_all=NC2_all, NC2_retain=NC2_retain,
        NC3=NC3,
        Delta_all=Delta_all, Delta_retain=Delta_retain, Delta_forget=Delta_forget,
        cos_all_min=cs_all[0], cos_all_max=cs_all[1], cos_all_avg=cs_all[2],
        cos_ret_min=cs_ret[0], cos_ret_max=cs_ret[1], cos_ret_avg=cs_ret[2],
        cos_for_min=cs_for[0], cos_for_max=cs_for[1], cos_for_avg=cs_for[2],
        CDNV_all=CDNV_all, CDNV_retain=CDNV_retain, CDNV_forget=CDNV_forget
    )
    return metrics