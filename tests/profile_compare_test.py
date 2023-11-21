import os
import time
import json
import logging
from src.data.datahandler import DataHandler
from src.model.batch_nmf import BatchNMF
from tests.factor_comparison import FactorComp
from src.utils import calculate_Q


if __name__ == "__main__":

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    t0 = time.time()
    for dataset in ["br"]:           # "br", "sl", "b", "w"
        for method in ["ls-nmf", "ws-nmf"]:     # "ls-nmf", "ws-nmf"
            for factors in range(6, 7):
                parallel = True
                optimized = True
                init_method = "col_means"           # default is column means, "kmeans", "cmeans"
                init_norm = True
                seed = 42
                models = 500
                if method == "ws-nmf":
                    converge_delta = 1.0 if dataset == "b" else 0.01
                    max_iterations = 5000
                    converge_n = 5
                else:
                    max_iterations = 50000
                    converge_delta = 0.01
                    converge_n = 10
                index_col = "Date"

                if dataset == "br":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"br{factors}f_residuals.txt")
                elif dataset == "b":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_con.txt")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_unc.txt")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Baltimore")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"b{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"b{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"b{factors}f_residuals.txt")
                elif dataset == "sl":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "StLouis")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"sl{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"sl{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"sl{factors}f_residuals.txt")
                elif dataset == "w":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "user_data", "wash_con_cleaned.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "user_data", "wash_unc_cleaned.csv")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Washington")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"w{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"w{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"w{factors}f_residuals.txt")
                    index_col = "obs_date"

                sn_threshold = 2.0

                dh = DataHandler(
                    input_path=input_file,
                    uncertainty_path=uncertainty_file,
                    index_col=index_col,
                    sn_threshold=sn_threshold
                )
                V = dh.input_data_processed
                U = dh.uncertainty_data_processed

                batch_nmf = BatchNMF(V=V, U=U, factors=factors, models=models, method=method, seed=seed, init_method=init_method,
                                     init_norm=init_norm, fuzziness=5.0, max_iter=max_iterations, converge_delta=converge_delta,
                                     converge_n=converge_n, parallel=parallel, optimized=optimized)
                t0 = time.time()
                batch_nmf.train()

                t1 = time.time()
                runtime = round(t1-t0, 2)
                print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")
                full_output_path = f"{dataset}-nmf-output-f{factors}.json"
                batch_nmf.save(output_name=full_output_path)

                profile_comparison = FactorComp(nmf_output_file=full_output_path, pmf_profile_file=pmf_profile_file,
                                                pmf_contribution_file=pmf_contribution_file, factors=factors,
                                                species=len(dh.features), residuals_path=pmf_residuals_file)
                pmf_q = calculate_Q(profile_comparison.pmf_residuals.values, dh.uncertainty_data_processed)
                profile_comparison.compare(PMF_Q=pmf_q)
                os.remove(path=full_output_path)

                run_type = "rust" if optimized else "py"

                run_key = f"{dataset}-{factors}-{method}"
                analysis_results = {
                    run_key:
                        {
                            "dataset": dataset,
                            "factors": factors,
                            "method": method,
                            "best_model": profile_comparison.best_model,
                            "factor_mapping": profile_comparison.factor_map,
                            "model_profiles": profile_comparison.nmf_epochs_dfs[profile_comparison.best_model]["H"].values.tolist(),
                            "model_contributions": profile_comparison.nmf_epochs_dfs[profile_comparison.best_model]["W"].values.tolist(),
                            "model_profile_r2": float(profile_comparison.best_factor_r_avg),
                            "model_contribution_r2": float(profile_comparison.best_contribution_r_avg)
                        }
                }
                current_results = {}
                analysis_file = "profile_compare_analysis.json"
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as json_file:
                        current_results = json.load(json_file)
                        if run_key in current_results.keys():
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