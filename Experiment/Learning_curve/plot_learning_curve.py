import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from pathlib import Path



import json
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1) Data processing / merge
# =========================================================

def _as_list(d, key, default=None):
    if default is None:
        default = []
    v = d.get(key, default)
    if v is None:
        return []
    if not isinstance(v, list):
        raise TypeError(f"Expected list for key='{key}', got {type(v)}")
    return v


def merge_series_by_iter(iter_x, iter_y, epoch_x, epoch_y, prefer_epoch=True):
    """
    Merge (iter_x, iter_y) with (epoch_x, epoch_y) on x=iteration.
    - Return sorted unique xs and aligned ys.
    - If duplicate x exists:
        prefer_epoch=True -> epoch_y overwrites iter_y at that x.
    """
    m = {}

    # iter points
    for x, y in zip(iter_x, iter_y):
        m[int(x)] = float(y)

    # epoch points (overwrite if prefer_epoch)
    for x, y in zip(epoch_x, epoch_y):
        x = int(x)
        if prefer_epoch or (x not in m):
            m[x] = float(y)

    xs = np.array(sorted(m.keys()), dtype=int)
    ys = np.array([m[x] for x in xs], dtype=float)
    return xs, ys

def downsample_iter_series(iter_x, *ys, k=2):
    """
    Downsample iter_x and aligned iter_y series together by step k.
    - Does NOT touch epoch arrays.
    - If iter_x is empty, returns as-is.
    """
    if k is None or k <= 1:
        return iter_x, ys
    if iter_x is None or len(iter_x) == 0:
        return iter_x, ys

    idx = slice(None, None, k)  # ::k
    iter_x_ds = list(np.array(iter_x, dtype=int)[idx])

    ys_ds = []
    for y in ys:
        if y is None or len(y) == 0:
            ys_ds.append(y)
            continue
        y_arr = np.array(y, dtype=float)
        if len(y_arr) != len(iter_x):
            raise ValueError(f"Downsample: len(y)={len(y_arr)} != len(iter_x)={len(iter_x)}")
        ys_ds.append(list(y_arr[idx]))
    return iter_x_ds, tuple(ys_ds)

def prepare_merged_forget_curves(history, iter_downsample=2):
    """
    Input: dict loaded from your json (model.history_log)
    Output:
      merged: dict of {metric_name: (xs, ys)} where xs are sorted merged iterations
      epoch_marks: dict with epoch ids, epoch iteration positions, and epoch metric values
      epoch_vlines: list of (epoch_id, x_iter) for epoch>=1 (for drawing vertical dashed lines)
    """

    # ---- base keys ----
    iter_x = _as_list(history, "iter")
    epoch_end_iter = _as_list(history, "epoch_end_iter")

    # ---- forget metrics from iter ----
    iter_forget_out = _as_list(history, "iter_forget_acc")
    iter_forget_lp  = _as_list(history, "iter_LP_forget_acc")
    iter_forget_ncc = _as_list(history, "iter_NCC_forget_acc")

    # ---- forget metrics from epoch-end ----
    # Prefer these if present (value-only arrays)
    epoch_forget_out = _as_list(history, "epoch_end_forget_acc")
    epoch_forget_lp  = _as_list(history, "epoch_end_LP_forget")
    epoch_forget_ncc = _as_list(history, "epoch_end_NCC_forget")

    # Defensive checks (same as your original)
    if len(epoch_end_iter) == 0:
        raise ValueError("epoch_end_iter is empty; cannot place epoch markers.")

    if len(iter_x) != len(iter_forget_out):
        if len(iter_x) != 0:
            raise ValueError(f"len(iter)={len(iter_x)} != len(iter_forget_acc)={len(iter_forget_out)}")
    if len(iter_x) != len(iter_forget_lp):
        if len(iter_x) != 0:
            raise ValueError(f"len(iter)={len(iter_x)} != len(iter_LP_forget_acc)={len(iter_forget_lp)}")
    if len(iter_x) != len(iter_forget_ncc):
        if len(iter_x) != 0:
            raise ValueError(f"len(iter)={len(iter_x)} != len(iter_NCC_forget_acc)={len(iter_forget_ncc)}")

    def _check_epoch_len(name, arr):
        if len(arr) != len(epoch_end_iter):
            raise ValueError(f"len({name})={len(arr)} != len(epoch_end_iter)={len(epoch_end_iter)}. "
                             f"Fix logging alignment first.")

    _check_epoch_len("epoch_end_forget_acc", epoch_forget_out)
    _check_epoch_len("epoch_end_LP_forget",  epoch_forget_lp)
    _check_epoch_len("epoch_end_NCC_forget", epoch_forget_ncc)


    if len(iter_x) > 0 and iter_downsample is not None and iter_downsample > 1:
        iter_x, (iter_forget_out, iter_forget_lp, iter_forget_ncc) = downsample_iter_series(
            iter_x,
            iter_forget_out, iter_forget_lp, iter_forget_ncc,
            k=iter_downsample
        )

    # ---- merge (epoch overwrites iter at same x) ----
    m_out_x, m_out_y = merge_series_by_iter(iter_x, iter_forget_out, epoch_end_iter, epoch_forget_out, prefer_epoch=True)
    m_lp_x,  m_lp_y  = merge_series_by_iter(iter_x, iter_forget_lp,  epoch_end_iter, epoch_forget_lp,  prefer_epoch=True)
    m_ncc_x, m_ncc_y = merge_series_by_iter(iter_x, iter_forget_ncc, epoch_end_iter, epoch_forget_ncc, prefer_epoch=True)

    merged = {
        "forget_output": (m_out_x, m_out_y),
        "forget_lp":     (m_lp_x,  m_lp_y),
        "forget_ncc":    (m_ncc_x, m_ncc_y),
    }

    # ---- epoch markers ----
    # epoch ids are 0..E where E=len(epoch_end_iter)-1
    epoch_ids = list(range(len(epoch_end_iter)))
    epoch_marks = {
        "epoch_ids": epoch_ids,
        "epoch_iter": np.array(epoch_end_iter, dtype=int),
        "forget_output": np.array(epoch_forget_out, dtype=float),
        "forget_lp":     np.array(epoch_forget_lp, dtype=float),
        "forget_ncc":    np.array(epoch_forget_ncc, dtype=float),
    }

    # ---- vertical lines for epoch1/2/3 ... (skip epoch0) ----
    epoch_vlines = []
    for e in epoch_ids:
        if e == 0:
            continue
        epoch_vlines.append((e, int(epoch_end_iter[e])))

    return merged, epoch_marks, epoch_vlines


# =========================================================
# 2) Plotting
# =========================================================

def plot_forget_curves_with_epoch_vlines(
    merged, epoch_marks, epoch_vlines,
    title="Forget curves (Output / LP / NCC)",
    out_png=None
):
    """
    Design goals:
      - merged curves: small markers
      - epoch points: slightly larger markers
      - vertical dashed lines for epoch1.. with labels above x-axis
      - grid on
      - avoid clutter: one legend entry per curve & per epoch-marker set (not duplicated)
    """
    #control tones and styles
    fs_title = 22
    fs_label = 20
    fs_tick  = 20
    fs_leg   = 20
    fs_epoch = 20
    fs_linewidth = 2.2


    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)

    # ---- merged lines (small markers) ----
    out_x, out_y = merged["forget_output"]
    lp_x,  lp_y  = merged["forget_lp"]
    ncc_x, ncc_y = merged["forget_ncc"]

    colors = {
        "output": "#457AB4", 
        "lp":     "#9E9898",  
        "ncc":    "#EA4337DA",  
    }

    ax.plot(out_x, out_y,
        linestyle="-", linewidth=fs_linewidth,
        color=colors["output"],      # 
        marker="o", markersize=6,
        label="Output")

    ax.plot(lp_x, lp_y,
            linestyle="--", linewidth=fs_linewidth,
            color=colors["lp"],     # 
            marker="s", markersize=6,
            label="Linear Probe")

    ax.plot(ncc_x, ncc_y,
            linestyle=":", linewidth=fs_linewidth,
            color=colors["ncc"],      # 
            marker="^", markersize=6,
            label="NCC")

    # ---- epoch overlay (slightly larger points) ----
    ex = epoch_marks["epoch_iter"]
    #ax.scatter(ex, epoch_marks["forget_output"], s=18, zorder=4, label="forget output (epoch)")
    #ax.scatter(ex, epoch_marks["forget_lp"],     s=18, zorder=4, label="forget LP (epoch)")
    #ax.scatter(ex, epoch_marks["forget_ncc"],    s=18, zorder=4, label="forget NCC acc (epoch)")
    ax.scatter(ex, epoch_marks["forget_output"], marker="o", color=colors["output"], s=64, zorder=8)
    ax.scatter(ex, epoch_marks["forget_lp"],     marker="s", color=colors["lp"], s=64, zorder=8)
    ax.scatter(ex, epoch_marks["forget_ncc"],    marker="^", color=colors["ncc"], s=64, zorder=8)

    # ---- vertical dashed lines + labels for epoch1.. ----
    # Put labels above axis; place label between previous and current epoch line if possible.
    ymin, ymax = ax.get_ylim()
    y_text = ymax + 0.25 * (ymax - ymin)  # will adjust after autoscale below

    # Need autoscale first so labels position is right
    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.02 * (ymax - ymin)

    # determine x-range
    x_min = min(out_x.min(), lp_x.min(), ncc_x.min())
    x_max = max(out_x.max(), lp_x.max(), ncc_x.max())

    # draw vlines and labels
    for (e, xline) in epoch_vlines:
        ax.axvline(x=xline, linestyle="--", linewidth=1.0, alpha=0.75)

    # label epoch1/2/3 on top between vlines
    # We'll label at midpoint between epoch_{e-1} and epoch_{e} (or just above line if previous missing)
    epoch_iter = epoch_marks["epoch_iter"]
    for (e, xline) in epoch_vlines:
        if e - 1 >= 0:
            x_left = int(epoch_iter[e - 1])
            x_pos = (x_left + xline) / 2.0
        else:
            x_pos = xline
        # clamp into visible region
        x_pos = max(x_min, min(x_max, x_pos))
        ax.text(x_pos, y_text, f"epoch{e}", ha="center", va="bottom", fontsize=fs_epoch)

    # expand top margin for labels
    ax.set_ylim(ymin, y_text + 0.05 * (ymax - ymin))

    # ---- cosmetics ----
    ax.set_title(title, fontsize=fs_title)
    ax.set_xlabel("Iteration", fontsize=fs_label)
    ax.set_ylabel("Accuracy", fontsize=fs_label)
    ax.tick_params(axis="both", labelsize=fs_tick)
    ax.legend(loc="center right", framealpha=0.9, fontsize=fs_leg)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    
    ax.margins(x=0.02)

    fig.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, bbox_inches="tight")
    return fig, ax


# =========================================================
# Main (edit only these paths/titles)
# =========================================================

if __name__ == "__main__":
    # --- YOU set these ---
    JSON_PATH = "checkpoints/random_label_iter_eval_learning_curve/cifar10_resnet18/0.json"    # e.g., "checkpoints/.../0.json"
    FIG_TITLE = ""
    OUT_PNG   = "plots/forget_curve_random_label_forget_0.png"       # or None if you don't want to save
    
    with open(JSON_PATH, "r") as f:
        history = json.load(f)

    merged, epoch_marks, epoch_vlines = prepare_merged_forget_curves(history, iter_downsample=3)

    # quick sanity print (optional)
    # print("epoch_vlines:", epoch_vlines)
    # print("merged output x head:", merged["forget_output"][0][:10])

    plot_forget_curves_with_epoch_vlines(
        merged=merged,
        epoch_marks=epoch_marks,
        epoch_vlines=epoch_vlines,
        title=FIG_TITLE,
        out_png=OUT_PNG
    )
    plt.show()
