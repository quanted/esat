import os
import time
import logging
from src.model.base_nmf import BaseSearch, BaseNMF
from src.data.datahandler import DataHandler
from tests.factor_comparison import FactorComp
from src.utils import calculate_Q


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


if __name__ == "__main__":


    t0 = time.time()
    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col
    )

    n_components = 6
    method = "mu"                   # "kl", "ls-nmf", "is", "euc", "gd"
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed
    seed = 42
    epochs = 50
    max_iterations = 50000
    converge_delta = 0.1
    converge_n = 1000

    a1 = time.time()
    bs = BaseSearch(n_components=n_components, method="kl-opt", V=V, U=U, seed=seed, epochs=epochs, max_iterations=max_iterations,
                    converge_delta=converge_delta, converge_n=converge_n)
    bs.optimized_train(parallel=True)
    a2 = time.time()
    # bs2 = BaseSearch(n_components=n_components, method="kl", V=V, U=U, seed=seed, epochs=epochs, max_iterations=max_iterations,
    #                 converge_delta=converge_delta, converge_n=converge_n)
    # bs2.parallel_train()
    # a3 = time.time()
    print(f"Rust - Q: {round(bs.results[bs.best_epoch]['Q'], 4)}, Steps: {bs.results[bs.best_epoch]['steps']}, Runtime: {round((a2-a1), 2)} sec(s)")
    # print(f"Python - Q: {round(bs2.results[bs2.best_epoch]['Q'], 4)}, Q: {bs2.results[bs2.best_epoch]['steps']}, Runtime: {round((a3-a2), 2)} sec(s)")

    full_output_path1 = "test-base-save-opt-01.json"
    bs.save(output_name=full_output_path1)

    # full_output_path2 = "test-base-save-opt-02.json"
    # bs2.save(output_name=full_output_path2)

    pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{n_components}f_profiles.txt")
    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                      f"br{n_components}f_residuals.txt")
    pmf_contributions_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                      f"br{n_components}f_contributions.txt")

    profile_comparison = FactorComp(nmf_output_file=full_output_path1, pmf_profile_file=pmf_file,
                                    pmf_contribution_file=pmf_contributions_file, factors=n_components,
                                    species=len(dh.features), residuals_path=pmf_residuals_file)
    pmf_q = calculate_Q(profile_comparison.pmf_residuals.values, dh.uncertainty_data_processed)
    profile_comparison.compare(PMF_Q=pmf_q)

    # profile_comparison2 = FactorComp(nmf_output=full_output_path2, pmf_output=pmf_file, factors=n_components,
    #                                 species=len(dh.features), residuals_path=pmf_residuals_file)
    # profile_comparison2.compare(PMF_Q=pmf_q)
    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")