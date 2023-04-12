import os
import time
import logging
from src.model.base_nmf import BaseSearch, BaseNMF
from src.data.datahandler import DataHandler


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

if __name__ == "__main__":

    t0 = time.time()
    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")

    # input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-con.csv")
    # uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-unc.csv")
    # output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "StLouis")

    # input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_con.txt")
    # uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_unc.txt")
    # output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Baltimore")

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col
    )

    n_components = 4
    method = "gd"  # "kl", "ls-nmf", "is", "euc"
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed
    seed = 42
    epochs = 10
    max_iterations = 1
    converge_delta = 0.1
    converge_n = 100

    bs = BaseSearch(n_components=n_components, method=method, V=V, U=U, seed=seed, epochs=epochs,
                    max_iterations=max_iterations,
                    converge_delta=converge_delta, converge_n=converge_n)
    bs.train()

    # full_output_path = "test-base-save-01.json"
    # bs.save(output_name=full_output_path)
    max_iterations2 = 20000
    bs2_results = []
    for result in bs.results:
        H = result["H"]
        W = result["W"]
        epoch = result["epoch"]
        _seed = result["seed"]
        _nmf = BaseNMF(
            n_components=n_components,
            method="mu",
            V=V,
            U=U,
            H=H,
            W=W,
            seed=_seed
        )
        _nmf.train(epoch=epoch, max_iterations=max_iterations2, converge_delta=converge_delta,
                   converge_n=converge_n)
        bs2_results.append({
            "epoch": epoch,
            "Q": float(_nmf.Qtrue),
            "steps": _nmf.converge_steps,
            "converged": _nmf.converged,
            "H": _nmf.H,
            "W": _nmf.W,
            "wh": _nmf.WH,
            "seed": int(_seed)
        })
    test = 1
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baltimore_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baltimore_{n_components}f_residuals.txt")
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baton-rouge_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baton-rouge_{n_components}f_residuals.txt")
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
    #                                   f"br{n_components}f_residuals.txt")
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"stlouis_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"stlouis_{n_components}f_residuals.txt")
    # profile_comparison = FactorComp(nmf_output=full_output_path, pmf_output=pmf_file, factors=n_components,
    #                                 species=len(dh.features), residuals_path=pmf_residuals_file)
    # pmf_q = calculate_Q(profile_comparison.pmf_residuals.values, dh.uncertainty_data_processed)
    # profile_comparison.compare(PMF_Q=pmf_q)

    t1 = time.time()
    print(f"Runtime: {round((t1 - t0) / 60, 2)} min(s)")
