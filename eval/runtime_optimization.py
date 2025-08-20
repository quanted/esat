import time
import numpy as np
import plotly.graph_objects as go

from esat_eval.simulator import Simulator
from esat.data.datahandler import DataHandler
from esat.model.sa import SA
from esat.model.batch_sa import BatchSA


rng = np.random.default_rng(42)

def benchmark_sa_on(sizes):

    results = []
    for size in sizes:
        feature = rng.integers(low=10, high=50)
        seed = rng.integers(low=0, high=500000)
        print(f"\nDataset: samples={size}, features={feature}, seed={seed}")
        sim = Simulator(
            seed=seed,
            factors_n=6,
            features_n=feature,
            samples_n=size,
            verbose=False
        )
        syn_input_df, syn_uncertainty_df = sim.get_data()
        data_handler = DataHandler.load_dataframe(input_df=syn_input_df, uncertainty_df=syn_uncertainty_df)
        V, U = data_handler.get_data()

        for use_gpu in [False, True]:
            print(f"  Running SA with GPU={'ON' if use_gpu else 'OFF'}")

            # Run and aggregate the results of the batch
            batch_sa = BatchSA(V=V, U=U, factors=6, models=20, method="ls-nmf", seed=seed, max_iter=50000,
                    converge_delta=0.1, converge_n=10,
                    verbose=False, use_gpu=use_gpu
            )
            start = time.time()
            _ = batch_sa.train()
            elapsed = time.time() - start
            mean_qtrue = np.mean([m.Qtrue for m in batch_sa.results])
            mean_qrobust = np.mean([m.Qrobust for m in batch_sa.results])

            results.append({
                "size": size,
                "features": feature,
                "seed": seed,
                "gpu": use_gpu,
                "time": elapsed,
                "q_true": mean_qtrue,
                "q_robust": mean_qrobust
            })

    # Calculate time differences for each size (non-GPU - GPU)
    # Prepare data for line plots
    sizes_sorted = sorted(set(r['size'] for r in results))
    gpu_times = [next(r['time'] for r in results if r['size'] == s and r['gpu']) for s in sizes_sorted]
    non_gpu_times = [next(r['time'] for r in results if r['size'] == s and not r['gpu']) for s in sizes_sorted]

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes_sorted,
        y=non_gpu_times,
        mode='lines+markers',
        name='Non-GPU'
    ))
    fig.add_trace(go.Scatter(
        x=sizes_sorted,
        y=gpu_times,
        mode='lines+markers',
        name='GPU'
    ))
    fig.update_layout(
        title="Runtime for SA by Dataset Size",
        xaxis_title="Dataset Size (samples)",
        yaxis_title="Runtime (seconds)",
        legend_title="Configuration"
    )
    fig.write_html('gpu_runtime_results.html')

    return results

if __name__ == "__main__":
    # Example configurations
    dataset_sizes = [1000, 2000, 3000]
    # dataset_sizes = np.arange(1000, 100000, 1000)
    use_gpu_flags = [False, True]

    results = benchmark_sa_on(dataset_sizes)

    print("\nSummary:")
    for r in results:
        print(f"Samples: {r['size']}, Features: {r['features']}, Seed: {r['seed']}, "
              f"GPU: {r['gpu']}, Time: {r['time']:.2f}s, Q(true): {r['q_true']:.4f}, Q(robust): {r['q_robust']:.4f}")