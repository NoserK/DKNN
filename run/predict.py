"""
predict.py  (adapted for the ABCD simulations)

Change vs. the original: the original predict() read a separate full
random-field file from ./Data/random_field/ and predicted the whole grid.
Our simulations have no such file, and we need predictions on the held-out
TEST split (which are exactly the unknown locations we care about). So this
version predicts on the test split of the sampled dataset and writes a
per-point result CSV (id, coodx, coody, target, predict) on the ORIGINAL
target scale, which main.py joins back for metric computation.

Everything about the model, scaling, positional embedding, and the
target-zeroing convention is unchanged.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")                       # headless-safe for batch runs
import matplotlib.pyplot as plt

from utils.utils import DIAGNOSIS
from utils.loaddataset import DataSet
from model.net import DKNN


def visualize_field_pre(filepath):
    """Scatter of true vs. predicted target over test coordinates.
       (The original reshaped to a 100x100 grid; the test split is a scattered
       subset, so we scatter instead.)"""
    data = pd.read_csv(filepath)
    fig = plt.figure(figsize=(12, 5))
    for i, col in enumerate(["target", "predict"]):
        ax = plt.subplot(1, 2, i + 1)
        sc = ax.scatter(data["coodx"], data["coody"], c=data[col], s=12)
        plt.colorbar(sc, ax=ax)
        ax.set_title(["true", "prediction"][i], fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath[:filepath.rfind("/")] + "/RFresult_visualize.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def predict(modelname, datapath, train_info, hidden_neurons, pe_weight, top_k,
            is_save_result=True):
    d_input, d_model, d_trend = hidden_neurons
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ##### read + scale the sampled dataset #####
    sampledata = pd.read_csv(datapath)
    datafile = datapath[datapath.rfind("/") + 1:]
    datafilename = datafile[0:datafile.rfind(".")]

    dataset = DataSet(sampledata)
    data_scaler = dataset.scaler_data()
    data_train_scaler = data_scaler["train"]
    data_test_scaler = data_scaler["test"]

    ##### known (observed) points = TRAIN split #####
    known_coods_scaler = torch.from_numpy(
        data_train_scaler.values[:, 1:3].astype(float)).to(torch.float32).to(device)
    known_feature_scaler = torch.from_numpy(
        data_train_scaler.values[:, 3:3 + d_input].astype(float)).to(torch.float32).to(device)

    ##### unknown (query) points = TEST split #####
    unknown_coods_scaler = torch.from_numpy(
        data_test_scaler.values[:, 1:3].astype(float)).to(torch.float32).to(device)
    unknown_feature_scaler = torch.from_numpy(
        data_test_scaler.values[:, 3:3 + d_input].astype(float)).to(torch.float32).to(device)

    ##### load the best model #####
    save_dir = "./results/" + modelname + "/" + datafilename + "/" + train_info
    net = DKNN(d_input=d_input, d_model=d_model, known_num=dataset.train_num,
               d_trend=d_trend, top_k=top_k, pe_weight=pe_weight)
    net.load_state_dict(torch.load(save_dir + "/checkpoint.pth",
                                   map_location=torch.device(device)))
    net.cal_pe_know(known_feature_scaler, known_coods_scaler)
    net.cal_pe_unknow(unknown_feature_scaler, unknown_coods_scaler)
    net.to(device)

    ##### predict on the TEST split #####
    test_loader = DataLoader(data_test_scaler.values, shuffle=False,
                             batch_size=500, drop_last=False)
    out_scaled = []
    with torch.no_grad():
        net.eval()
        for i in test_loader:
            i = i.to(torch.float32)
            input_feature = i[:, 3:3 + d_input].to(device)
            input_feature[:, -1] = 0                       # target is unknown
            input_coods = i[:, 1:3].to(device)
            input_pe = net.pe_unknow[i[:, 0].type(torch.long)]
            output, _ = net(input_coods, input_feature, input_pe,
                            known_coods_scaler, known_feature_scaler)
            out_scaled.extend(output.cpu().detach().numpy())

    # inverse-transform to the original target scale
    pred_inv = dataset.scaler_label.inverse_transform(
        np.array(out_scaled).reshape(-1, 1)).reshape(-1)

    # original ids / coords / true target, in the same (test) order
    test_orig = dataset.test_data.reset_index(drop=True)
    result = pd.DataFrame({
        "id":     test_orig.values[:, 0],
        "coodx":  test_orig.values[:, 1],
        "coody":  test_orig.values[:, 2],
        "target": test_orig.values[:, -1].astype(float),
        "predict": pred_inv,
    })

    diag = DIAGNOSIS(pred_inv.reshape(-1, 1),
                     test_orig.values[:, -1].astype(float).reshape(-1, 1))
    rmse_inv, mse_inv, mae_inv, mape_inv = diag.get()
    print("TEST MAE/RMSE/MAPE: {:.4f}/{:.4f}/{:.2f}%".format(
        mae_inv, rmse_inv, mape_inv * 100))

    result_path = save_dir + "/RFresult.csv"
    if is_save_result:
        os.makedirs(save_dir, exist_ok=True)
        result.to_csv(result_path, index=False)

    return [mae_inv, rmse_inv, mape_inv], result_path
