from src.model.model import NMFModel
from src.model.sklearn_nmf import NMF
from src.data.datahandler import DataHandler
from tests.factor_comparison import FactorComp
import numpy as np
import os
import time
import json


if __name__ == "__main__":

    t0 = time.time()

    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")

    index_col = "Date"
    n_components = 4
    max_iter = 100000
    tol = 1e-8

    seed = 1
    init = 'random'
    solver = 'mu'
    q_loss = True
    n_prior_errors = 100
    error_tol = 0.01
    beta_loss = 2

    epochs = 10

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col
    )

    rng = np.random.default_rng(seed)

    X = dh.input_data_processed
    U = dh.uncertainty_data_processed
    results = []

    best_model = None
    best_q = float("inf")

    H_list = []
    W_list = []
    a_W = 4.1
    a_H = 4.1

    for e in range(epochs):
        t2 = time.time()
        e_seed = rng.integers(low=0, high=1e5)
        nmf = NMF(n_components=n_components, beta_loss=beta_loss, init=init, tol=tol, max_iter=max_iter, solver=solver,
                  random_state=e_seed, alpha_W=a_W, alpha_H=a_H, verbose=0, q_loss=q_loss,
                  n_prior_errors=n_prior_errors, error_tol=error_tol)
        # 4.5, 4.5 = 0.9625
        W = nmf.fit_transform(X=X, U=U)
        H = nmf.components_
        Q = nmf.reconstruction_err_
        t3 = time.time()
        results.append({
            "epoch": e,
            "Q": nmf.Qrobust.astype(float),
            "steps": nmf.n_iter_,
            "converged": (nmf.n_iter_ < max_iter),
            "H": nmf.H.astype(float).tolist(),
            "W": nmf.W.astype(float).tolist(),
            "wh": nmf.WH.astype(float).tolist()
        })
        H_list.append(nmf.H)
        W_list.append(nmf.W)
        # print(f"Epoch: {e}, Q: {Q}, Iterations: {nmf.n_iter_}, Seed: {e_seed}, Runtime: {round(t3-t2, 2)} sec")

        if Q < best_q:
            best_q = Q
            best_model = e

    t1 = time.time()
    print(f"Best Epoch: {best_model}, Best Q: {best_q}, Runtime: {round((t1-t0)/60, 3)} min(s)")
    t4 = time.time()
    tf_max_iterations = 20000

    lr_initial = 1e-1
    lr_decay_steps = 100
    lr_decay_rate = 0.85
    converge_diff = 0.01
    converge_iter = 50

    index_col = "Date"

    model = NMFModel(
        dh=dh,
        epochs=epochs,
        n_components=n_components,
        max_iterations=tf_max_iterations,
        seed=seed,
        lr_initial=lr_initial,
        lr_decay_steps=lr_decay_steps,
        lr_decay_rate=lr_decay_rate,
        converge_diff=converge_diff,
        converge_iter=converge_iter,
        initial_H=H_list,
        initial_W=W_list
    )
    model.fit()
    model.print_results()

    full_output_path = "test-save-04.json"
    model.save(output_name=full_output_path)
    t5 = time.time()
    print(f"Runtime: {round((t5 - t4) / 60, 3)} min(s)")

    pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", "baton-rouge_4f_profiles.txt")
    profile_comparison = FactorComp(nmf_output=full_output_path, pmf_output=pmf_file, factors=4, species=41)
    profile_comparison.compare()
