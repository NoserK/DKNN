"""
main.py  (adapted to batch-run DKNN on the ABCD simulation settings)

For each setting and replicate, DKNN is trained TWICE (target = var1, then
var2), because DKNN is univariate. The two point-prediction vectors are
combined into a bivariate prediction and scored with metrics_common.metrics,
so the numbers line up exactly with the CDE-AP table.

DKNN is a point interpolator with no native predictive distribution, so by
default only MSE / MAD / MD are reported and CRPS / COV95 are left blank
(NaN) — leave those cells empty in the comparison table.

Assumptions that depend on run/predict.py (not provided):
  * predict() writes a result CSV at `result_path` containing, for each point,
    a prediction column and either an `id` column or coordinate columns.
  * predictions are on the ORIGINAL target scale (consistent with the
    `best_inverse` MAE/RMSE the training loop reports).
If your column names differ, adjust extract_test_predictions() — it raises a
clear error listing the available columns.

Usage:
    python main.py                      # all settings, 50 replicates
    python main.py setting_C 50
    python main.py all 50 --top_k 300
"""
import os, sys, argparse
import numpy as np
import pandas as pd

from run.train import train
from run.predict import predict

from utils.loaddataset import convert_to_dknn_format
from metrics_common import metrics

# ───────────────────────── configuration ───────────────────────────────
DATA_ROOT = "/Users/zeminjiang/Documents/CV/DKNN-main/Data/dataset"                 # where setting_A/ ... setting_D/ live (cwd)
DKNN_DATA_DIR = "./Data/dataset"  # where converted CSVs are written
MODELNAME = "DKNN"

# hyperparameters (sample defaults; tune as needed for n_train≈1080)
BATCH_SIZE = 128
LR = 1e-4
MODEL_DIM = 256
TREND_DIM = 16
PE_WEIGHT = 0.8
TOP_K = 400
LOSS_TYPE = "rmse"
OPTIM_TYPE = "adam"


def extract_test_predictions(result_path, test_frame):
    """Read predict()'s output and return predictions aligned to test_frame
       row order (original target scale)."""
    res = pd.read_csv(result_path)
    lc = {c.lower(): c for c in res.columns}

    pred_candidates = ["predict", "pre", "pred", "prediction", "target_pre",
                       "pre_target", "z_pre", "pre_inverse", "target_pred",
                       "y_pre", "pred_target", "estimate", "pre_z"]
    pred_col = next((lc[c] for c in pred_candidates if c in lc), None)
    if pred_col is None:
        raise ValueError(
            f"[extract_test_predictions] No prediction column found in "
            f"{result_path}. Available columns: {list(res.columns)}. "
            f"Add the correct name to `pred_candidates`.")

    if "id" in lc:
        merged = test_frame.merge(res[[lc["id"], pred_col]],
                                  left_on="id", right_on=lc["id"], how="left")
        preds = merged[pred_col].values.astype(float)
    else:
        xcol = lc.get("coodx", lc.get("x"))
        ycol = lc.get("coody", lc.get("y"))
        if xcol is None or ycol is None:
            raise ValueError(
                f"[extract_test_predictions] No `id` and no coordinate columns "
                f"in {result_path}. Available columns: {list(res.columns)}.")
        tf = test_frame.copy()
        tf["kx"] = tf["coodx"].round(6); tf["ky"] = tf["coody"].round(6)
        r = res.copy()
        r["kx"] = r[xcol].round(6); r["ky"] = r[ycol].round(6)
        merged = tf.merge(r[["kx", "ky", pred_col]], on=["kx", "ky"], how="left")
        preds = merged[pred_col].values.astype(float)

    if np.isnan(preds).any():
        raise ValueError(
            "[extract_test_predictions] Some test points did not match a "
            "prediction (NaN after join). Check the join key / prediction scale.")
    return preds


def run_one_replicate(tag, sim, top_k, gaussian_uq=False, n_samples=500, seed=0):
    y_cols = ["var1", "var2"]
    preds = {}
    rmse_inv = {}
    for target_col in y_cols:
        datapath, n_aux, test_frame = convert_to_dknn_format(
            tag, sim, target_col, data_root=DATA_ROOT, out_dir=DKNN_DATA_DIR)
        hidden_neuron = [n_aux + 1, MODEL_DIM, TREND_DIM]

        train_info, min_loss, best_epoch, best_inverse = train(
            MODELNAME, datapath, BATCH_SIZE, LR, hidden_neuron,
            PE_WEIGHT, top_k, loss_type=LOSS_TYPE, optim_type=OPTIM_TYPE,
            if_summary=False, if_save_model=True)

        _, result_path = predict(
            MODELNAME, datapath, train_info, hidden_neuron,
            pe_weight=PE_WEIGHT, top_k=top_k, is_save_result=True)

        preds[target_col] = extract_test_predictions(result_path, test_frame)
        rmse_inv[target_col] = float(best_inverse[1])   # per-coord RMSE (orig scale)

    # bivariate point predictions in test order (build() preserved order)
    te = pd.read_csv(f"{DATA_ROOT}/{tag}/testing_data/2D_{tag}_1200_{sim}-test.csv")
    y = te[y_cols].values
    mu = np.column_stack([preds["var1"], preds["var2"]])

    if gaussian_uq:
        # OPTIONAL, discouraged: constant Gaussian using per-coord RMSE as sigma.
        # This uses test-set RMSE, so the resulting coverage is optimistic —
        # report it only as a rough sanity value, not as DKNN's coverage.
        sigma = np.array([rmse_inv["var1"], rmse_inv["var2"]])
        rng = np.random.RandomState(seed + sim)
        z = rng.standard_normal((n_samples, len(y), 2))
        samples = mu[None] + sigma[None, None, :] * z
        m = metrics(samples, mu, y, mode="joint")
        m.update(metrics(samples, mu, y, mode="marginal"))
    else:
        # honest default: point predictor -> only MSE/MAD/MD; distributional NaN
        m = metrics(None, mu, y, mode="joint")
        m.update(metrics(None, mu, y, mode="marginal"))
        m["CRPS"] = np.nan
        m["COV95"] = np.nan

    m["model"] = "DKNN"
    m["sim"] = sim
    return m


def run_setting(tag, n_sim, top_k, gaussian_uq=False):
    rows = []
    for sim in range(1, n_sim + 1):
        rows.append(run_one_replicate(tag, sim, top_k, gaussian_uq=gaussian_uq))
        print(f"[{tag}] replicate {sim}/{n_sim} done")
    df = pd.DataFrame(rows)
    summ = df.drop(columns=["sim"]).groupby("model").mean(numeric_only=True).round(4)
    print(f"\n=== {tag}: DKNN (means over {n_sim} replicates) ===")
    print(summ.to_string())
    out = f"dknn_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}\n")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="all",
                    help="setting_A | setting_B | setting_C | setting_D | all")
    ap.add_argument("n_sim", nargs="?", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--gaussian_uq", action="store_true",
                    help="(discouraged) add a constant-Gaussian predictive so "
                         "CRPS/COV95 are non-empty; coverage will be optimistic")
    args = ap.parse_args()

    tags = (["setting_A", "setting_B", "setting_C", "setting_D"]
            if args.tag == "all" else [args.tag])
    for tag in tags:
        run_setting(tag, args.n_sim, args.top_k, gaussian_uq=args.gaussian_uq)
