import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "..\\src")

import time
import json
import logging
from  esat.data.datahandler import DataHandler
from  esat.model.batch_sa import BatchSA


if __name__ == "__main__":

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    parallel = True
    optimized = True
    t0 = time.time()
    for dataset in ["sl"]:           # "br", "sl", "b"
        for method in ["ws-nmf"]:     # "ls-nmf", "ws-nmf"
            for factors in range(4, 11):
                init_method = "col_means"           # default is column means, "kmeans", "cmeans"
                init_norm = True
                seed = 42
                models = 100
                if method == "ws-nmf":
                    converge_delta = 0.1 if dataset == "b" else 0.01
                    max_iterations = 30000
                    converge_n = 50
                else:
                    max_iterations = 50000
                    converge_delta = 0.001
                    converge_n = 100
                index_col = "Date"

                if dataset == "br":
                    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-BatonRouge-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data",
                                                    "Dataset-BatonRouge-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "esat", "output", "BatonRouge")
                elif dataset == "b":
                    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-Baltimore_con.txt")
                    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-Baltimore_unc.txt")
                    output_path = os.path.join("D:\\", "projects", "esat", "output", "Baltimore")
                elif dataset == "sl":
                    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-StLouis-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-StLouis-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "esat", "output", "StLouis")

                sn_threshold = 2.0

                dh = DataHandler(
                    input_path=input_file,
                    uncertainty_path=uncertainty_file,
                    index_col=index_col,
                    sn_threshold=sn_threshold
                )
                V = dh.input_data_processed
                U = dh.uncertainty_data_processed

                batch_sa = BatchSA(V=V, U=U, factors=factors, models=models, method=method, seed=seed,
                                   init_method=init_method, init_norm=init_norm, fuzziness=5.0,
                                   max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n,
                                   parallel=parallel, optimized=optimized)
                t0 = time.time()
                batch_sa.train()

                t1 = time.time()
                runtime = round(t1-t0, 2)
                print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")
                q_true = []
                q_robust = []
                for sa in batch_sa.results:
                    if sa is not None:
                        q_true.append(sa.Qtrue)
                        q_robust.append(sa.Qrobust)

                run_key = f"{dataset}-{factors}"
                analysis_results = {
                    run_key:
                        {
                            "dataset": dataset,
                            "factors": factors,
                            f"{method}-Q(True)": q_true,
                            f"{method}-Q(Robust)": q_robust,
                }
                        }
                current_results = {}
                analysis_file = "loss_analysis_2024.json"
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
                logging.info(f"Completed method: {method}, factors: {factors}, dataset: {dataset}")
