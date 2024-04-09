import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "..\\src")

import time
import logging
import numpy as np
from  esat.model.batch_sa import BatchSA
from  esat.data.datahandler import DataHandler


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


if __name__ == "__main__":

    t0 = time.time()
    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-BatonRouge-unc.csv")

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        index_col=index_col
    )

    factors = 6
    V = dh.input_data_processed.astype(np.float64)
    U = dh.uncertainty_data_processed.astype(np.float64)
    seed = 42
    models = 10
    max_iterations = 20000
    converge_delta = 0.01
    converge_n = 100

    t0 = time.time()
    batch1 = BatchSA(V=V, U=U, factors=factors, models=models, method="ls-nmf", seed=seed, optimized=False)
    batch1.train()
    t1 = time.time()
    batch2 = BatchSA(V=V, U=U, factors=factors, models=models, method="ls-nmf", seed=seed, optimized=True)
    batch2.train()
    t2 = time.time()
    batch3 = BatchSA(V=V, U=U, factors=factors, models=models, method="ws-nmf", seed=seed, optimized=False)
    batch3.train()
    t3 = time.time()
    batch4 = BatchSA(V=V, U=U, factors=factors, models=models, method="ws-nmf", seed=seed, optimized=True)
    batch4.train()
    t4 = time.time()
    print(f"Rust Opt Test - LS-NMF, Py Runtime: {round(t1 - t0, 2)} sec(s), Rust Runtime: {round(t2 - t1, 2)} sec(s)")
    print(f"Rust Opt Test - WS-NMF, Py Runtime: {round(t3 - t2, 2)} sec(s), Rust Runtime: {round(t4 - t3, 2)} sec(s)")
