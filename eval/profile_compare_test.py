import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "..\\src")

import time
import json
import logging
from esat.data.datahandler import DataHandler
from esat.model.batch_sa import BatchSA
from eval.factor_comparison import FactorComp
from esat.metrics import calculate_Q, q_loss


if __name__ == "__main__":

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    completed_solutions = []
    analysis_file = os.path.join("results", "pc_analysis_2024_2.json")
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as a_file:
            completed = json.load(a_file)
            completed_solutions = list(completed.keys())

    t0 = time.time()
    for dataset in ["br"]:           # "br", "sl", "b"
        for method in ["ls-nmf"]:     # "ls-nmf", "ws-nmf"
            for factors in range(6, 7):
                run_key = f"{dataset}-{factors}-{method}"
                # if run_key in completed_solutions:
                #     print(f"{run_key} already completed.")
                #     continue

                optimized = True
                parallel = True
                init_method = "col_means"
                init_norm = True
                seed = 4
                models = 5
                max_iterations = 50000 if method == "ls-nmf" else 20000
                converge_delta = 0.01 if method == 'ws-nmf' else 0.001
                converge_n = 50
                index_col = "Date"

                if dataset == "br":
                    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-BatonRouge-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-BatonRouge-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "esat", "output", "BatonRouge")
                    pmf_profile_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test", f"br{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test", f"br{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test",
                                                      f"br{factors}f_residuals.txt")
                elif dataset == "b":
                    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-Baltimore_con.txt")
                    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-Baltimore_unc.txt")
                    output_path = os.path.join("D:\\", "projects", "esat", "output", "Baltimore")
                    pmf_profile_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test", f"b{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test", f"b{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test",
                                                      f"b{factors}f_residuals.txt")
                elif dataset == "sl":
                    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-StLouis-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-StLouis-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "esat", "output", "StLouis")
                    pmf_profile_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test", f"sl{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test", f"sl{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "esat", "data", "factor_test",
                                                      f"sl{factors}f_residuals.txt")

                sn_threshold = 2.0

                dh = DataHandler(
                    input_path=input_file,
                    uncertainty_path=uncertainty_file,
                    index_col=index_col,
                    sn_threshold=sn_threshold
                )
                V, U = dh.get_data()

                batch_sa = BatchSA(V=V, U=U, factors=factors, models=models, method=method, seed=seed,
                                   init_method=init_method, init_norm=init_norm, fuzziness=5.0,
                                   max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n,
                                   parallel=parallel, optimized=optimized)
                t0 = time.time()
                batch_sa.train()

                t1 = time.time()
                runtime = round(t1-t0, 2)
                print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")

                profile_comparison = FactorComp.load_pmf_output(factors=factors, input_df=dh.input_data,
                                                                uncertainty_df=dh.uncertainty_data,
                                                                pmf_profile_file=pmf_profile_file,
                                                                pmf_contribution_file=pmf_contribution_file,
                                                                batch_sa=batch_sa)
                profile_comparison.compare()
                pmf_Q = calculate_Q(profile_comparison.input_df.to_numpy() - profile_comparison.base_V_estimate.to_numpy(),
                                    profile_comparison.uncertainty_df.to_numpy())
                run_type = "rust" if optimized else "py"

                analysis_results = {
                    run_key:
                        {
                            "dataset": dataset,
                            "factors": factors,
                            "method": method,
                            "Q(True)": float(profile_comparison.sa_Q[profile_comparison.best_model]),
                            "PMF(Q)": float(pmf_Q),
                            "best_model": profile_comparison.best_model,
                            "factor_mapping": profile_comparison.factor_map,
                            # "model_profiles": profile_comparison.sa_model_dfs[profile_comparison.best_model]["H"].values.tolist(),
                            # "model_contributions": profile_comparison.sa_model_dfs[profile_comparison.best_model]["W"].values.tolist(),
                            "model_profile_r2": float(profile_comparison.best_factor_r_avg),
                            "model_contribution_r2": float(profile_comparison.best_contribution_r_avg),
                            f"runtime(sec)_{run_type}": runtime
                        }
                }
                current_results = {}

                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as json_file:
                        current_results = json.load(json_file)
                        if run_key in current_results.keys():
                            print(f"Previous profile R2: {analysis_results[run_key]['model_profile_r2']}, new "
                                  f"profile R2: {float(profile_comparison.best_factor_r_avg)}")
                            if analysis_results[run_key]['model_profile_r2'] < float(profile_comparison.best_factor_r_avg):
                                print(f"Replacing existing results with better correlated model")
                                for k, v in analysis_results[run_key].items():
                                    current_results[run_key].update({k: v})
                        else:
                            current_results[run_key] = analysis_results[run_key]
                else:
                    current_results = analysis_results
                with open(analysis_file, 'w') as json_file:
                    current_results = dict(sorted(current_results.items()))
                    json.dump(current_results, json_file)
                logging.info(f"Completed method: {method}, type: {run_type}, factors: {factors}, dataset: {dataset}")