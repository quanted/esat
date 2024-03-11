import os
import time
import json
import logging
from src.data.datahandler import DataHandler
from src.model.batch_sa import BatchSA


if __name__ == "__main__":
    """
    Runtime comparison analysis compares how long it takes for the ESAT algorithms to converge with a loss value that 
    is within 10% of the optimal PMF5 solution. 
    
    The target Q from PMF5 has been recorded and used to set the Q target range, the BatchSA is executed with the 
    default hyper-parameters (convergence criteria). If the optimal ESAT solution has a lower Q value than the target
    range,  converge_delta is increased by (mid+high)/2, and if the solution has a higher Q value than the target 
    range, converge_delta is decreased by (mid+low)/2. Where mid = 1e-3, high=10.0, and low = 1e-5, converge_n is fixed
    at 10 for ls-nmf and 5 for ws-nmf. 
    """

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    q_targets_file = os.path.join("..", "results", "q_analysis.json")
    q_targets = {}
    with open(q_targets_file, 'r') as q_file:
        q_ =  json.load(q_file)
        for k, v in q_.items():
            q_targets[k] = v["Q(pmf)"]

    t0 = time.time()
    for dataset in ["br", "sl", "b"]:           # "br", "sl", "b", "w"
        init_method = "col_means"  # default is column means, "kmeans", "cmeans"
        init_norm = True
        seed = 42
        models = 10
        index_col = "Date"

        if dataset == "br":
            input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
            uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
            output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")
            pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                            f"br{factors}f_profiles.txt")
            pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                 f"br{factors}f_contributions.txt")
            pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                              f"br{factors}f_residuals.txt")
        elif dataset == "b":
            input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_con.txt")
            uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_unc.txt")
            output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Baltimore")
            pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                            f"b{factors}f_profiles.txt")
            pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                 f"b{factors}f_contributions.txt")
            pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                              f"b{factors}f_residuals.txt")
        elif dataset == "sl":
            input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-con.csv")
            uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-unc.csv")
            output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "StLouis")
            pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                            f"sl{factors}f_profiles.txt")
            pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                 f"sl{factors}f_contributions.txt")
            pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                              f"sl{factors}f_residuals.txt")

        sn_threshold = 2.0

        dh = DataHandler(
            input_path=input_file,
            uncertainty_path=uncertainty_file,
            index_col=index_col,
            sn_threshold=sn_threshold
        )
        V = dh.input_data_processed
        U = dh.uncertainty_data_processed

        for method in ["ls-nmf", "ws-nmf"]:     # "ls-nmf", "ws-nmf"
            if method == "ws-nmf":
                max_iterations = 2000
            else:
                max_iterations = 50000
            converge_delta = 1e-3
            converge_n = 10 if method == "ls-nmf" else 5
            for factors in range(3, 11):
                run_key = f"{dataset}-{factors}"
                target_min = q_targets[run_key] * 0.9
                target_max = q_targets[run_key] * 1.1
                conv_mid = converge_delta
                conv_high = 10.0
                conv_low = 1e-5
                conv_found = False
                current_q = 0.0
                max_search = 50
                search_i = 0
                for (parallel, optimized) in ((True, True), (True, False)):
                    while search_i < max_search or (target_min < current_q < target_max):
                        search_i += 1

                        batch_sa = BatchSA(V=V, U=U, factors=factors, models=models, method=method, seed=seed,
                                           init_method=init_method, init_norm=init_norm, fuzziness=5.0,
                                           max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n,
                                           parallel=parallel, optimized=optimized)
                        t0 = time.time()
                        batch_sa.train()

                        t1 = time.time()
                        runtime = round(t1-t0, 2)
                        current_q = batch_sa.results[batch_sa.best_model].Qtrue
                        if current_q < target_min:
                            conv_mid = (conv_high + conv_mid)/2.0
                            conv_low = conv_mid
                        else:
                            conv_mid = (conv_low + conv_mid)/2.0
                            conv_high = conv_mid

                        if conv_found is True:
                            break
                    conv_found = True
                    run_type = "rust" if optimized else "py"
                    print(f"Runtime: {round((t1 - t0) / 60, 2)} min(s)")

                    analysis_results = {
                        run_key:
                            {
                                "dataset": dataset,
                                "factors": factors,
                                f"{method}-{run_type}-runtime": runtime,
                            }
                    }
                    current_results = {}
                    analysis_file = "type_runtime_analysis.json"
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