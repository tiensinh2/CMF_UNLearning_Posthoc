#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run this script in the directory that contains the `evaluations/` folder.

It will:
1) Read JSON files ONLY by your explicit settings (no recursion).
2) Extract metrics:
   - test_retain_acc, test_forget_acc
   - lp_acc_test_retain, lp_acc_test_forget
   - ncc_acc_test_retain, ncc_acc_test_forget
3) Keep single-class and multi-class buckets SEPARATE (bucket="1"/"3"/"10"/"20").
4) Write:
   - raw_results.csv        (one row per JSON)
   - summary_results.csv    (mean/var per method/dataset/arch/bucket)
   - report.txt             (missing files, bad json, missing keys)
   - table_{arch}.tex       (paper-style table; single vs multi shown separately)
"""

import os
import json
import math
from typing import Dict, List, Tuple, Optional

import pandas as pd


# =========================
# 0) Config (YOU edit here)
# =========================

EVAL_ROOT = "evaluations"  # you run in the directory containing this folder

ARCHES = ["resnet18"]  # arch set (you will tune)

METHODS = [
    #"pre_train",
    #"retrain",
    #"random_label",
    #"random_label",
    #"random_label_CMF_RemoveFC",
    #"salun",
    #"salun_CMF_RemoveFC",
    #"grad_descent",
    #"grad_ascent_descent",
    #"grad_ascent_descent_CMF_RemoveFC",
    "tarun",
    #"tarun_CMF_RemoveFC",
    #"scrub",
    #"scrub_CMF_RemoveFC",

    
    # add more if you have folders:
    # "pre_train",
    # "retain_only_retrain",
]

# dataset -> bucket -> labels (labels must match the JSON filename WITHOUT ".json")
# Buckets are kept separate, so single vs multi are naturally separated.
DATASET_BUCKETS: Dict[str, Dict[str, List[str]]] = {
    "cifar10": {
        "1": [str(i) for i in range(10)],
        "3": [
            "0,1,2",
            "3,4,5",
            "6,7,8",
            "0,5,9",
            "2,4,8",
        ],
    },
    "cifar100": {
        "1": ["0", "1", "2", "3", "5"],
        "10": [
            "3,15,19,21,31,38,42,43,88,97",
            "47,52,54,56,59,62,70,82,92,96",
            "5,20,22,25,39,40,84,86,87,94",
            "8,13,41,48,59,69,81,85,89,90",
            "1,4,30,32,55,67,72,73,91,95",
        ],
    },
    "tinyimagenet": {
        "1": ["2", "3", "5", "7", "9"],
        "20": [
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
            "20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39",
            "40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59",
            "60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79",
            "80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99",
        ],
    },
}

# metrics to extract from JSON (top-level keys)
METRICS = [
    "test_retain_acc",
    "test_forget_acc",
    "lp_acc_test_retain",
    "lp_acc_test_forget",
    "ncc_acc_test_retain",
    "ncc_acc_test_forget",
]


# =========================
# 1) Helpers
# =========================

def build_path(method: str, dataset: str, arch: str, classes: str) -> str:
    # evaluations/{method}/{dataset}_{arch}/{classes}.json
    return os.path.join(EVAL_ROOT, method, f"{dataset}_{arch}", f"{classes}.json")


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"__json_error__": True}


def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def nanmean(vals: List[float]) -> float:
    xs = [v for v in vals if not math.isnan(v)]
    return float("nan") if len(xs) == 0 else sum(xs) / len(xs)


def nanvar(vals: List[float], ddof: int = 1) -> float:
    xs = [v for v in vals if not math.isnan(v)]
    n = len(xs)
    if n == 0:
        return float("nan")
    if n - ddof <= 0:
        return float("nan")
    mu = sum(xs) / n
    return sum((v - mu) ** 2 for v in xs) / (n - ddof)


def fmt_pct(x: float, decimals: int = 2) -> str:
    if math.isnan(x):
        return "--"
    return f"{100*x:.{decimals}f}"


def method_display_name(m: str) -> str:
    mapping = {
        "random_label_CMF_RemoveFC": "Random-label with CMF",
        "salun_CMF_RemoveFC": "Salun with CMF",
        "grad_ascent_descent_CMF_RemoveFC": "Grad Ascent/Descent with CMF",
        "pre_train": "Original",
        "retain_only_retrain": "Retain-only Retrain",
    }
    return mapping.get(m, m)


def dataset_display_name(d: str) -> str:
    if d == "tinyimagenet":
        return "Tiny-ImageNet"
    return d.upper()


def expected_multi_bucket(dataset: str) -> str:
    # used for LaTeX table columns (single=1 vs multi)
    return {"cifar10": "3", "cifar100": "10", "tinyimagenet": "20"}[dataset]


# =========================
# 2) Collect raw (NO recursion, strict by labels)
#    IMPORTANT: classes loop is the INNERMOST loop (as you requested).
# =========================

def collect_raw_rows() -> Tuple[pd.DataFrame, List[str], List[str], List[Tuple[str, str]]]:
    missing_files: List[str] = []
    bad_json: List[str] = []
    missing_keys: List[Tuple[str, str]] = []  # (path, missing_key)

    rows = []

    # arch outermost (your request)
    for arch in ARCHES:
        for method in METHODS:
            for dataset, buckets in DATASET_BUCKETS.items():
                for bucket, labels in buckets.items():

                    # ---- INNERMOST: label/classes ----
                    for classes in labels:
                        path = build_path(method, dataset, arch, classes)
                        obj = load_json(path)

                        if obj is None:
                            missing_files.append(path)
                            continue
                        if obj.get("__json_error__", False):
                            bad_json.append(path)
                            continue

                        row = {
                            "arch": arch,
                            "method": method,
                            "dataset": dataset,
                            "bucket": bucket,   # "1"/"3"/"10"/"20" -> keeps single vs multi separated
                            "classes": classes,
                            "path": path,
                        }

                        for k in METRICS:
                            if k not in obj:
                                missing_keys.append((path, k))
                            row[k] = safe_float(obj.get(k, float("nan")))

                        rows.append(row)

    df = pd.DataFrame(rows)
    return df, missing_files, bad_json, missing_keys


# =========================
# 3) Summary (mean/var per method/dataset/arch/bucket)
# =========================

def summarize(df_raw: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    group_cols = ["arch", "method", "dataset", "bucket"]
    out_rows = []

    for keys, g in df_raw.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))

        for k in METRICS:
            vals = [safe_float(x) for x in g[k].tolist()]
            m = nanmean(vals)
            v = nanvar(vals, ddof=ddof)
            s = math.sqrt(v) if not math.isnan(v) else float("nan")

            row[f"{k}_mean"] = m
            row[f"{k}_var"] = v
            row[f"{k}_std"] = s

        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(group_cols).reset_index(drop=True)

def fmt_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    if math.isnan(mean):
        return "--"
    if math.isnan(std):
        return f"{100*mean:.{decimals}f}"
    return f"{100*mean:.{decimals}f} ± {100*std:.{decimals}f}"

def fmt_mean_std_latex(mean: float, std: float, decimals: int = 2) -> str:
    """
    Format as LaTeX math: $xx.xx \pm yy.yy$
    """
    if math.isnan(mean):
        return "--"
    if math.isnan(std):
        return f"${100*mean:.{decimals}f}$"
    return f"${100*mean:.{decimals}f}{{\\scriptstyle \\pm {100*std:.{decimals}f}}}$"

# =========================
# 4) LaTeX table for a given arch
#    - Single vs Multi are shown separately (bucket 1 vs bucket multi)
#    - Each method has 3 rows: Full / LP / NCC
# =========================
def fmt_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    if math.isnan(mean):
        return "--"
    if math.isnan(std):
        return f"{100*mean:.{decimals}f}"
    return f"{100*mean:.{decimals}f} ± {100*std:.{decimals}f}"


def latex_table_for_arch(df_sum: pd.DataFrame, arch: str, outfile: str) -> None:
    """
    Paper-style LaTeX table with proper booktabs group rules:
      - \toprule, \midrule, \bottomrule
      - dataset-group \cmidrule(lr){...}
      - bucket-group  \cmidrule(lr){...}
    Values are formatted as: $mean{\scriptstyle \pm std}$.
    """

    dfa = df_sum[df_sum["arch"] == arch].copy()
    if dfa.empty:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f"% No data for arch={arch}\n")
        return

    # Keep a stable dataset order (and only those present)
    datasets = [d for d in ["cifar10", "cifar100", "tinyimagenet"] if d in dfa["dataset"].unique()]

    # For column headers: single bucket is always "1", multi depends on dataset
    multi_bucket = {"cifar10": "3", "cifar100": "10", "tinyimagenet": "20"}

    # Row types: (label, retain_mean_col, forget_mean_col)
    row_types = [
        ("Full Model", "test_retain_acc_mean", "test_forget_acc_mean"),
        ("Feature Mapping (LP)", "lp_acc_test_retain_mean", "lp_acc_test_forget_mean"),
        ("NC Accuracy", "ncc_acc_test_retain_mean", "ncc_acc_test_forget_mean"),
    ]

    def get_mean_std(method: str, dataset: str, bucket: str, mean_col: str) -> Tuple[float, float]:
        # std col name convention: *_mean -> *_std
        std_col = mean_col.replace("_mean", "_std")
        sub = dfa[
            (dfa["method"] == method) &
            (dfa["dataset"] == dataset) &
            (dfa["bucket"] == bucket)
        ]
        if sub.empty:
            return float("nan"), float("nan")
        r = sub.iloc[0]
        return float(r.get(mean_col, float("nan"))), float(r.get(std_col, float("nan")))

    def fmt_mean_std_latex(mean: float, std: float, decimals: int = 2) -> str:
        # $93.45{\scriptstyle \pm 0.59}$
        if math.isnan(mean):
            return "--"
        if math.isnan(std):
            return f"${100*mean:.{decimals}f}$"
        return f"${100*mean:.{decimals}f}{{\\scriptstyle \\pm {100*std:.{decimals}f}}}$"

    def dataset_display_name(d: str) -> str:
        return "Tiny-ImageNet" if d == "tinyimagenet" else d.upper().replace("CIFAR10", "CIFAR-10").replace("CIFAR100", "CIFAR-100")

    # ---------- Build automatic cmidrule ranges ----------
    # Column layout is: 1) Method, 2) Layers, then for each dataset: 4 columns
    # So dataset i (0-indexed) occupies columns:
    #   start = 3 + 4*i, end = start + 3
    dataset_ranges = []
    bucket_ranges = []
    for i, _d in enumerate(datasets):
        ds_start = 3 + 4 * i
        ds_end = ds_start + 3
        dataset_ranges.append((ds_start, ds_end))               # e.g., (3,6), (7,10), (11,14)
        bucket_ranges.append((ds_start, ds_start + 1))          # single: 2 cols
        bucket_ranges.append((ds_start + 2, ds_start + 3))      # multi : 2 cols

    def cmidrules(ranges: List[Tuple[int, int]]) -> str:
        # join: \cmidrule(lr){3-6}\cmidrule(lr){7-10}...
        return "".join([rf"\cmidrule(lr){{{a}-{b}}}" for a, b in ranges])

    # ---------- LaTeX assembly ----------
    lines = []
    lines.append(r"\begingroup")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\small")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Evaluation of MU methods on \textbf{{{arch}}}. Values are mean accuracies (\%). Buckets (single vs multi) are separated.}}")
    lines.append(rf"\label{{tab:eval_{arch.replace('/','_')}}}")
    lines.append(r"\vspace{-0.12in}")

    col_spec = "l l " + " ".join(["cccc" for _ in datasets])
    lines.append(r"\resizebox{1.0\linewidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row 1: dataset names
    header1 = [r"\multirow{3}{*}{\textbf{Method}}", r"\multirow{3}{*}{\textbf{Layers Evaluated}}"]
    for d in datasets:
        header1.append(r"\multicolumn{4}{c}{\textbf{" + dataset_display_name(d) + r"}}")
    lines.append(" & ".join(header1) + r" \\")
    # Dataset group rules
    lines.append(cmidrules(dataset_ranges))

    # Header row 2: bucket labels
    header2 = ["", ""]
    for d in datasets:
        header2.append(r"\multicolumn{2}{c}{\textbf{1}}")
        header2.append(r"\multicolumn{2}{c}{\textbf{" + multi_bucket[d] + r"}}")
    lines.append(" & ".join(header2) + r" \\")
    # Bucket group rules
    lines.append(cmidrules(bucket_ranges))

    # Header row 3: Retain/Forget
    header3 = ["", ""]
    for _ in datasets:
        header3 += [r"\textbf{Retain}", r"\textbf{Forget}", r"\textbf{Retain}", r"\textbf{Forget}"]
    lines.append(" & ".join(header3) + r" \\")
    lines.append(r"\midrule")

    # Body
    for method in METHODS:
        mname = method_display_name(method)

        for i, (rtype, col_ret, col_for) in enumerate(row_types):
            row = []
            if i == 0:
                row.append(r"\multirow{3}{*}{" + mname + "}")
            else:
                row.append("")
            row.append(rtype)

            for d in datasets:
                b1 = "1"
                bm = multi_bucket[d]

                m11, s11 = get_mean_std(method, d, b1, col_ret)
                m12, s12 = get_mean_std(method, d, b1, col_for)
                m21, s21 = get_mean_std(method, d, bm, col_ret)
                m22, s22 = get_mean_std(method, d, bm, col_for)

                row += [
                    fmt_mean_std_latex(m11, s11),
                    fmt_mean_std_latex(m12, s12),
                    fmt_mean_std_latex(m21, s21),
                    fmt_mean_std_latex(m22, s22),
                ]

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    # Replace last \midrule with \bottomrule (clean booktabs style)
    if lines and lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append(r"\endgroup")

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# 5) Main
# =========================

def main():
    # 1) raw
    df_raw, missing_files, bad_json, missing_keys = collect_raw_rows()
    df_raw.to_csv("raw_results.csv", index=False)

    # 2) summary: mean + var (sample variance ddof=1)
    df_sum = summarize(df_raw, ddof=1)
    df_sum.to_csv("summary_results.csv", index=False)

    # 3) latex per arch (arch already outermost in collection; here we also loop arch)
    for arch in ARCHES:
        latex_table_for_arch(df_sum, arch, outfile=f"table_{arch}.tex")

    # 4) report
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("=== Raw rows collected ===\n")
        f.write(f"{len(df_raw)}\n\n")

        f.write("=== Summary rows ===\n")
        f.write(f"{len(df_sum)}\n\n")

        f.write("=== Missing JSON files (not found) ===\n")
        for p in missing_files:
            f.write(p + "\n")
        f.write("\n")

        f.write("=== Bad JSON files (decode error) ===\n")
        for p in bad_json:
            f.write(p + "\n")
        f.write("\n")

        f.write("=== Missing keys (path, key) ===\n")
        for p, k in missing_keys:
            f.write(f"{p}\t{k}\n")
        f.write("\n")

    print("[OK] wrote raw_results.csv")
    print("[OK] wrote summary_results.csv")
    print("[OK] wrote report.txt")
    print("[OK] wrote latex tables:", ", ".join([f"table_{a}.tex" for a in ARCHES]))


if __name__ == "__main__":
    main()
