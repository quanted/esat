import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "..\\src")

import time
import json
import logging
import plotly.graph_objects as go

from esat.data.datahandler import DataHandler
from esat.model.batch_sa import BatchSA


def plot_runtime_results_plotly(analysis_file_path, output_dir):
    if os.path.exists(analysis_file_path):
        with open(analysis_file_path, 'r') as json_file:
            results = json.load(json_file)

        # Prepare data for plotting
        plot_data = {}

        for key, value in results.items():
            dataset = value["dataset"]
            if dataset not in plot_data:
                plot_data[dataset] = {}

            for method in ["ls-nmf", "ws-nmf"]:  # Explicitly handle known methods
                runtime_key = f"{method}-runtime"
                if runtime_key in value:
                    if method not in plot_data[dataset]:
                        plot_data[dataset][method] = []
                    plot_data[dataset][method].append((value["factors"], value[runtime_key]))

        # Create Plotly figure
        fig = go.Figure()

        for dataset, method_data in plot_data.items():
            for method, data in method_data.items():
                data.sort(key=lambda x: x[0])  # Sort by factors
                if data:  # Ensure data is not empty
                    factors, runtimes = zip(*data)
                    fig.add_trace(go.Scatter(
                        x=factors,
                        y=runtimes,
                        mode='lines+markers',
                        name=f"{dataset} - {method}"
                    ))

        # Update layout
        fig.update_layout(
            title="Runtime Analysis",
            xaxis_title="Factors",
            yaxis_title="Runtime (seconds)",
            legend_title="Dataset - Method",
            template="plotly_white"
        )

        # Save the plot
        output_file = os.path.join(output_dir, "runtime_results_plot.html")
        fig.write_html(output_file)


if __name__ == "__main__":
    """
    This script runs a runtime analysis for different datasets and factorization algorithms, evaluating the performance 
    of different compiling optimization levels.
    """

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    project_dir = os.path.join("D:\\", "git", "esat")

    analysis_file_path = os.path.join(project_dir, "eval", "results", "runtime_analysis.json")
    plot_file_path = os.path.join(project_dir, "eval", "results", "runtime_results_plot.html")
    # delete the analysis file if it exists
    if os.path.exists(analysis_file_path):
        os.remove(analysis_file_path)
    if os.path.exists(analysis_file_path):
        os.remove(plot_file_path)

    t0 = time.time()
    for dataset in ["b"]:           # "br", "sl", "b", "w"
        init_method = "col_means"
        seed = 42
        models = 10
        index_col = "Date"

        if dataset == "br":
            input_file = os.path.join(project_dir, "data", "Dataset-BatonRouge-con.csv")
            uncertainty_file = os.path.join(project_dir, "data", "Dataset-BatonRouge-unc.csv")
            output_path = os.path.join(project_dir, "data", "output", "BatonRouge")
        elif dataset == "b":
            input_file = os.path.join(project_dir, "data", "Dataset-Baltimore_con.txt")
            uncertainty_file = os.path.join(project_dir, "data", "Dataset-Baltimore_unc.txt")
            output_path = os.path.join(project_dir, "data", "output", "Baltimore")
        elif dataset == "sl":
            input_file = os.path.join(project_dir, "data", "Dataset-StLouis-con.csv")
            uncertainty_file = os.path.join(project_dir, "data", "Dataset-StLouis-unc.csv")
            output_path = os.path.join(project_dir, "data", "output", "StLouis")

        dh = DataHandler(
            input_path=input_file,
            uncertainty_path=uncertainty_file,
            index_col=index_col
        )
        V, U = dh.get_data()

        for method in ["ls-nmf", "ws-nmf"]:
            converge_delta = 0.1
            converge_n = 25
            if method == "ws-nmf":
                max_iterations = 10000
                converge_delta = 1.0
            else:
                max_iterations = 50000

            for factors in range(3, 9):
                run_key = f"{dataset}-{factors}"
                batch_sa = BatchSA(V=V, U=U, factors=factors, models=models, method=method, seed=seed, init_method=init_method, max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n)
                t0 = time.time()
                batch_sa.train()

                t1 = time.time()
                runtime = round(t1 - t0, 2)
                current_q = batch_sa.results[batch_sa.best_model].Qtrue
                for k, s in enumerate(batch_sa.results):
                    if s is not None:
                        if s.Qtrue < current_q:
                            current_q = s.Qtrue
                print(f"Runtime: {round((t1 - t0) / 60, 2)} min(s)")

                analysis_results = {
                    run_key:
                        {
                            "dataset": dataset,
                            "factors": factors,
                            f"{method}-runtime": runtime,
                            f"{method}-Q": float(current_q)
                        }
                }
                current_results = {}
                if os.path.exists(analysis_file_path):
                    with open(analysis_file_path, 'r') as json_file:
                        current_results = json.load(json_file)
                        if run_key in current_results.keys():
                            for k, v in analysis_results[run_key].items():
                                current_results[run_key].update({k: v})
                        else:
                            current_results[run_key] = analysis_results[run_key]
                else:
                    current_results = analysis_results
                with open(analysis_file_path, 'w') as json_file:
                    current_results = dict(sorted(current_results.items()))
                    json.dump(current_results, json_file)
                logging.info(f"Completed method: {method}, factors: {factors}, dataset: {dataset}, Runtime: {round((t1 - t0) / 60, 2)} min(s)")
    output_dir = os.path.join("D:\\", "git", "esat", "eval", "results")
    plot_runtime_results_plotly(analysis_file_path, output_dir)