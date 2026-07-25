"""
dknn_tuning.py
==============
Diagnostic + hyperparameter grid search for DKNN on settings A, B, D
(where the authors' defaults, chosen for a 400-point dataset, fail).

Place at the DKNN repo root next to main.py. Uses the same adapted
run/train.py, run/predict.py, utils/loaddataset.py as main.py.

Protocol (matches Appendix app:dknn in the paper):
  * Grid: lr x loss_type x top_k over TUNING_SIMS replicates only.
  * Diagnostics per config: MSE, MSE/Var(y) ratio (≈1 means the fit
    degenerated to a near-constant prediction), correlation(pred, target).
  * Selection: lowest mean joint MSE across tuning replicates.
  * IMPORTANT: replicates used here must be EXCLUDED from the evaluation
    run (evaluate on replicates 6..50 in main.py, or generate 5 extra
    replicates for tuning). Tuning and evaluating on the same replicates
    would bias the comparison in DKNN's favour.

Usage:
    python dknn_tuning.py setting_B
    python dknn_tuning.py setting_A --sims 1 2 3
    python dknn_tuning.py all              # A, B, D
"""
import argparse, itertools, time
import numpy as np
import pandas as pd

from run.train import train
from run.predict import predict
from utils.loaddataset import convert_to_dknn_format

# ── grid (keep small: each cell trains 2 nets per replicate) ────────────
LR_GRID    = [1e-3, 3e-4, 1e-4]
LOSS_GRID  = ["rmse", "mse", "mae"]
TOPK_GRID  = [50, 100, 400]
TUNING_SIMS = [1, 2, 3, 4, 5]          # exclude these from evaluation!

MODELNAME = "DKNN"
BATCH_SIZE, MODEL_DIM, TREND_DIM, PE_WEIGHT = 128, 256, 16, 0.8
DATA_ROOT, DKNN_DATA_DIR = "/Users/zeminjiang/Documents/CV/DKNN-main/Data/dataset", "./Data/dataset"


def fit_predict_one(tag, sim, target_col, lr, loss_type, top_k):
    datapath, n_aux, test_frame = convert_to_dknn_format(
        tag, sim, target_col, data_root=DATA_ROOT, out_dir=DKNN_DATA_DIR)
    hidden = [n_aux + 1, MODEL_DIM, TREND_DIM]
    train_info, min_loss, best_epoch, best_inv = train(
        MODELNAME, datapath, BATCH_SIZE, lr, hidden, PE_WEIGHT, top_k,
        loss_type=loss_type, optim_type="adam",
        if_summary=False, if_save_model=True)
    _, result_path = predict(MODELNAME, datapath, train_info, hidden,
                             pe_weight=PE_WEIGHT, top_k=top_k,
                             is_save_result=True)
    res = pd.read_csv(result_path)
    return res["predict"].values.astype(float), res["target"].values.astype(float), best_epoch


def eval_config(tag, sims, lr, loss_type, top_k):
    per_sim = []
    for sim in sims:
        mus, ys = [], []
        epochs = []
        for target_col in ("var1", "var2"):
            p, t, be = fit_predict_one(tag, sim, target_col, lr, loss_type, top_k)
            mus.append(p); ys.append(t); epochs.append(be)
        mu = np.column_stack(mus); y = np.column_stack(ys)
        mse = float(((mu - y) ** 2).mean())
        var_y = float(y.var(axis=0).mean())
        corr = float(np.mean([np.corrcoef(mu[:, j], y[:, j])[0, 1] for j in range(2)]))
        per_sim.append({"sim": sim, "MSE": mse, "MSE_over_VarY": mse / var_y,
                        "corr": corr, "best_epoch_mean": float(np.mean(epochs))})
    df = pd.DataFrame(per_sim)
    return {
        "lr": lr, "loss": loss_type, "top_k": top_k,
        "MSE": df["MSE"].mean(),
        "MSE_over_VarY": df["MSE_over_VarY"].mean(),
        "corr": df["corr"].mean(),
        "best_epoch_mean": df["best_epoch_mean"].mean(),
    }


def run_setting(tag, sims):
    print(f"\n#### {tag}: grid search over {len(LR_GRID)*len(LOSS_GRID)*len(TOPK_GRID)} "
          f"configs x {len(sims)} replicates x 2 coords ####")
    rows = []
    for lr, loss_type, top_k in itertools.product(LR_GRID, LOSS_GRID, TOPK_GRID):
        t0 = time.time()
        r = eval_config(tag, sims, lr, loss_type, top_k)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        flag = ""
        if r["MSE_over_VarY"] > 0.9:
            flag = "  <-- ~Var(y): degenerate (near-constant) fit"
        print(f"lr={lr:<7g} loss={loss_type:<5s} top_k={top_k:<4d} "
              f"MSE={r['MSE']:<12.4f} MSE/Var(y)={r['MSE_over_VarY']:.3f} "
              f"corr={r['corr']:.3f} epoch={r['best_epoch_mean']:.0f} "
              f"({r['seconds']}s){flag}")
    df = pd.DataFrame(rows).sort_values("MSE")
    out = f"dknn_tuning_{tag}.csv"
    df.to_csv(out, index=False)
    best = df.iloc[0]
    print(f"\nBEST for {tag}: lr={best.lr}, loss={best.loss}, top_k={int(best.top_k)} "
          f"(MSE={best.MSE:.4f}, MSE/Var(y)={best.MSE_over_VarY:.3f})")
    print(f"Full grid saved to {out}")
    print("Reminder: run the evaluation (main.py) on replicates NOT in "
          f"{sims}, with the selected configuration.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="all",
                    help="setting_A | setting_B | setting_D | all")
    ap.add_argument("--sims", nargs="*", type=int, default=TUNING_SIMS)
    args = ap.parse_args()
    tags = ["setting_A", "setting_B", "setting_D"] if args.tag == "all" else [args.tag]
    for t in tags:
        run_setting(t, args.sims)
