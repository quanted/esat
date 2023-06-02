import os
import time
import json
import logging
from src.data.datahandler import DataHandler
from src.model.base_nmf import BaseSearch
from tests.factor_comparison import FactorComp
from src.utils import calculate_Q


if __name__ == "__main__":
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    t0 = time.time()
    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=None,
        index_col=index_col
    )

    # Base model paramters

    min_components = 11  # min number of factors to compare
    max_components = 12
    method1 = "ls-nmf"  # minimization algorithm: 'euc' multiplicative update - frobenius, 'ls-nmf' least squares NMF, 'ws-nmf' weighted semi-nmf
    seed = 42  # randomization seed
    epochs = 20  # number of models to create
    max_iterations = 20000  # max number of iterations to run for multiplicative update models
    converge_delta = 0.01  # the amount of change between iterations for a multiplicative model considered converged
    converge_n = 100  # the number of iterations required with a loss change of less than converge_delta for the model to be considered converged

    run_all = False
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed

    # Run models and write output files
    output_path = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test")

    if run_all:
        for n_component in range(min_components, max_components + 1):
            base = BaseSearch(n_components=n_component, method=method1, V=V, U=U, seed=seed, epochs=epochs,
                              max_iterations=max_iterations, converge_delta=converge_delta, converge_n=converge_n)
            base.parallel_train()
            output_file = f"nmf-br{n_component}-output.json"
            base.save(output_name=output_file, output_path=output_path)

        t1 = time.time()
        print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")

    results = []

    for n_component in range(min_components, max_components+1):
        nmf_file = f"nmf-br{n_component}-output.json"
        nmf_output_file = os.path.join(output_path, nmf_file)
        pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{n_component}f_profiles.txt")
        pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{n_component}f_contributions.txt")
        pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                          f"br{n_component}f_residuals.txt")
        profile_comparison = FactorComp(nmf_output_file=nmf_output_file, pmf_profile_file=pmf_profile_file,
                                        pmf_contribution_file=pmf_contribution_file, factors=n_component,
                                        species=len(dh.features), residuals_path=pmf_residuals_file)
        pmf_q = calculate_Q(profile_comparison.pmf_residuals.values, dh.uncertainty_data_processed)

        profile_comparison.compare(PMF_Q=pmf_q)
        best_model = profile_comparison.best_model

        results = {
            "factors": n_component,
            "nmf-Q": profile_comparison.nmf_Q[best_model],
            "pmf-Q": float(pmf_q),
            "R2 Avg": float(profile_comparison.best_avg_r)
        }

        profiling_results_file = os.path.join(output_path, f"br{n_component}f_profiling_results.json")
        with open(profiling_results_file, 'w') as json_file:
            json.dump(results, json_file)
