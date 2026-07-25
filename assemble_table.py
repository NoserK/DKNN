"""
assemble_table.py
=================
Read the per-method, per-setting result CSVs produced by the comparison
programs and emit a single LaTeX comparison table per setting, with
structural blanks (\\na) where a method does not produce a given metric.

Inputs (default names; edit SOURCES if yours differ):
    cde_ap_{tag}.csv          # CDE-AP (your main run)   -> MSE MAD MD CRPS COV95 [width]
    deepkriging_{tag}.csv     # from deepkriging_comparison.py
    geocp_{tag}.csv           # from geoconformal_comparison.py (GeoCP)
    geocp_{tag}_simcp.csv     # from geoconformal_comparison.py --geosimcp
    dknn_{tag}.csv            # from main.py (adapted DKNN)

Each CSV is expected to have per-replicate rows and (optionally) a `model`
column. Columns are matched case-insensitively via METRIC_SYNONYMS, so the
slightly different names across programs (e.g. 'width' vs 'Width_joint',
'COV95' vs 'COV95_joint') are all handled.

The emitted table uses \\best{...} for the best value per column and \\na for
missing metrics. Define these in your preamble (already present in
experiments_section.tex):
    \\newcommand{\\best}[1]{\\textbf{#1}}
    \\newcommand{\\na}{\\textemdash}

Usage:
    python assemble_table.py setting_C
    python assemble_table.py all
"""
import argparse, os, warnings
import numpy as np
import pandas as pd

# ── which files/rows feed which table row (edit to match your filenames) ──
SOURCES = [
    {"label": "CDE-AP (ours)",       "path": "cde_ap_{tag}.csv",      "model": None},
    {"label": "DeepKriging",         "path": "deepkriging_{tag}.csv", "model": "DeepKriging-standard"},
    {"label": "GeoCP + PointNet",    "path": "geocp_{tag}.csv",       "model": None},
    {"label": "GeoSIMCP + PointNet", "path": "geocp_{tag}_simcp.csv", "model": None},
    {"label": "DKNN",                "path": "dknn_{tag}.csv",        "model": None},
]

METRIC_ORDER = ["MSE", "MAD", "MD", "CRPS", "COV95", "Width"]
LOWER_BETTER = {"MSE", "MAD", "MD", "CRPS", "Width"}   # COV95: closest to 0.95
NOMINAL_COV = 0.95

METRIC_SYNONYMS = {
    "MSE":   ["MSE", "mse", "mse_joint"],
    "MAD":   ["MAD", "mad", "mad_joint"],
    "MD":    ["MD", "md"],
    "CRPS":  ["CRPS", "crps"],
    "COV95": ["COV95", "COV95_joint", "cov95", "coverage", "coverage_probability"],
    "Width": ["width", "Width", "Width_joint", "mean_width", "mean_width_finite",
              "interval_width"],
}


def _pick(available_lc, names):
    for n in names:
        if n.lower() in available_lc:
            return available_lc[n.lower()]
    return None


def load_row(src, tag):
    path = src["path"].format(tag=tag)
    if not os.path.exists(path):
        warnings.warn(f"missing {path}; skipping '{src['label']}'")
        return None
    df = pd.read_csv(path)
    if src.get("model") and "model" in df.columns:
        sub = df[df["model"] == src["model"]]
        if len(sub) == 0:
            warnings.warn(f"model '{src['model']}' not found in {path}; "
                          f"available: {sorted(df['model'].unique())}; skipping")
            return None
        df = sub
    num = df.mean(numeric_only=True)
    available_lc = {c.lower(): c for c in num.index}
    row = {"Method": src["label"]}
    for m in METRIC_ORDER:
        c = _pick(available_lc, METRIC_SYNONYMS[m])
        row[m] = float(num[c]) if (c is not None and not pd.isna(num[c])) else np.nan
    return row


def build(tag):
    rows = []
    for s in SOURCES:
        r = load_row(s, tag)
        if r is not None:
            rows.append(r)
    if not rows:
        raise SystemExit(f"No result files found for {tag}. Check filenames/paths.")
    df = pd.DataFrame(rows).set_index("Method")
    return df[METRIC_ORDER]


def best_index_per_metric(df):
    best = {}
    for m in METRIC_ORDER:
        col = df[m].dropna()
        if col.empty:
            best[m] = None
        elif m == "COV95":
            best[m] = (col - NOMINAL_COV).abs().idxmin()
        else:
            best[m] = col.idxmin()
    return best


def to_latex(df, tag):
    best = best_index_per_metric(df)

    def fmt(val, metric, method):
        if pd.isna(val):
            return r"\na"
        s = f"{val:.3f}"
        return r"\best{" + s + "}" if best[metric] == method else s

    setting = tag.replace("setting_", "")
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Head-to-head comparison on Setting~" + setting +
               r", means over replicates. Best per column in bold; "
               r"\na{} marks a metric the method does not natively produce.}")
    out.append(r"\label{tab:comparison_" + tag + r"}")
    out.append(r"\small")
    out.append(r"\begin{tabular}{l" + "c" * len(METRIC_ORDER) + r"}")
    out.append(r"\toprule")
    out.append("Method & " + " & ".join(METRIC_ORDER) + r" \\")
    out.append(r"\midrule")
    for method, r in df.iterrows():
        cells = [fmt(r[m], m, method) for m in METRIC_ORDER]
        out.append(method + " & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def run(tag):
    df = build(tag)
    print(f"\n=== Comparison summary: {tag} ===")
    print(df.round(4).to_string())
    tex = to_latex(df, tag)
    out_path = f"comparison_{tag}.tex"
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"\nLaTeX written to {out_path}\n")
    print(tex)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="all",
                    help="setting_A | setting_B | setting_C | setting_D | all")
    args = ap.parse_args()
    tags = (["setting_A", "setting_B", "setting_C", "setting_D"]
            if args.tag == "all" else [args.tag])
    for t in tags:
        try:
            run(t)
        except SystemExit as e:
            print(e)
