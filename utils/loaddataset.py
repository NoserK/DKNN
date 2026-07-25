"""
loaddataset.py  (adapted for the ABCD bivariate spatial simulations)

Changes vs. the original DKNN loaddataset.py:
  1. DataSet is preserved almost verbatim; the ONLY change is a guard so it
     also works when a setting has no auxiliary variables (Setting C), where
     the StandardScaler would otherwise be handed a zero-column array.
  2. Added convert_to_dknn_format(): turns our (x, y, cov..., var1, var2)
     train/test CSVs into DKNN's expected single-target format
     (id, coodx, coody, aux..., target, dataset) for one chosen target
     coordinate, writes it to Data/dataset/, and returns the test frame so
     predictions can be joined back for scoring.

DKNN is UNIVARIATE (one `target`), so we convert twice per replicate — once
with target=var1 and once with target=var2 — and recombine in main.py.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataSet():
    ##### read and load dataset #####
    def __init__(self, data):
        super(DataSet, self).__init__()
        self.data = data
        self.train_data, self.test_data = self.load_dataset(self.data)
        self.train_num = len(self.train_data)

    def scaler_data(self):
        ##### scale dataset #####
        self.scaler_coods = MinMaxScaler(feature_range=(0, 1))   # coordinates
        self.scaler_features = StandardScaler()                  # auxiliary vars
        self.scaler_label = StandardScaler()                     # target

        data_train_scaler = self.train_data.copy()
        data_test_scaler = self.test_data.copy()

        # reset index (keep original id in the index, then renumber column 0)
        data_train_scaler.index = data_train_scaler.values[:, 0]
        data_test_scaler.index = data_test_scaler.values[:, 0]
        data_train_scaler.iloc[:, 0] = range(0, len(self.train_data))
        data_test_scaler.iloc[:, 0] = range(0, len(self.test_data))

        ##### scale coods #####
        data_train_scaler.iloc[:, 1:3] = self.scaler_coods.fit_transform(self.train_data.values[:, 1:3])
        data_test_scaler.iloc[:, 1:3] = self.scaler_coods.transform(self.test_data.values[:, 1:3])

        ##### scale auxiliary variables (guarded for the no-auxiliary case) #####
        n_aux = data_train_scaler.shape[1] - 4   # cols: id, x, y, [aux...], target
        if n_aux > 0:
            data_train_scaler.iloc[:, 3:-1] = self.scaler_features.fit_transform(self.train_data.values[:, 3:-1])
            data_test_scaler.iloc[:, 3:-1] = self.scaler_features.transform(self.test_data.values[:, 3:-1])

        ##### scale target variable #####
        data_train_scaler.iloc[:, -1] = self.scaler_label.fit_transform(self.train_data.values[:, -1].reshape(-1, 1))
        data_test_scaler.iloc[:, -1] = self.scaler_label.transform(self.test_data.values[:, -1].reshape(-1, 1))

        return {'train': data_train_scaler, 'test': data_test_scaler}

    def load_dataset(self, all_data):
        ##### load dataset #####
        if 'trend' in all_data.columns:
            all_data = all_data.drop('trend', axis=1, inplace=False)
        train_data = all_data[all_data['dataset'] == 'train']
        test_data = all_data[all_data['dataset'] == 'test']
        train_data = train_data.drop('dataset', axis=1, inplace=False)
        test_data = test_data.drop('dataset', axis=1, inplace=False)

        return train_data, test_data

    def get_data(self):
        return self.train_data, self.test_data


# ─────────────────────────────────────────────────────────────────────────
#  Conversion: our ABCD format  ->  DKNN single-target format
# ─────────────────────────────────────────────────────────────────────────
def convert_to_dknn_format(tag, sim, target_col,
                           data_root=".", out_dir="./Data/dataset"):
    """
    Read our train/test CSVs for (setting `tag`, replicate `sim`) and write a
    single DKNN-format CSV whose target is `target_col` (e.g. 'var1' or 'var2').

    Column order written (required by DataSet):
        id, coodx, coody, [cov1..covK], target, dataset

    Returns
    -------
    out_path : str
        Path of the written DKNN-format CSV (pass this as `datapath`).
    n_aux : int
        Number of auxiliary (covariate) columns -> hidden_neuron[0] = n_aux + 1.
    test_frame : DataFrame
        Columns [id, coodx, coody, target] for the test rows, in test order,
        used to join predictions back for metric computation.
    """
    tr = pd.read_csv(f"{data_root}/{tag}/training_data/2D_{tag}_1200_{sim}-train.csv")
    te = pd.read_csv(f"{data_root}/{tag}/testing_data/2D_{tag}_1200_{sim}-test.csv")
    cov_cols = [c for c in tr.columns if c.startswith("cov")]

    def build(df, split):
        out = pd.DataFrame()
        out["coodx"] = df["x"].values
        out["coody"] = df["y"].values
        for c in cov_cols:
            out[c] = df[c].values
        out["target"] = df[target_col].values
        out["dataset"] = split
        return out

    combined = pd.concat([build(tr, "train"), build(te, "test")], ignore_index=True)
    combined.insert(0, "id", range(len(combined)))   # unique, stable ids

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_sim{sim}_{target_col}.csv")
    combined.to_csv(out_path, index=False)

    test_frame = (combined[combined["dataset"] == "test"]
                  [["id", "coodx", "coody", "target"]]
                  .reset_index(drop=True))
    return out_path, len(cov_cols), test_frame
