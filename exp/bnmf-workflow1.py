import json
import os

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

import matplotlib.pyplot as plt
from tqdm import tqdm

from factor_catalog import BatchFactorCatalog
from esat.model.batch_sa import BatchSA
from esat_eval.simulator import Simulator
from esat.data.datahandler import DataHandler


def compute_reconstruction_error(W, H, V, U):
    if isinstance(W, np.ndarray):
        diff = (V - np.dot(W, H)) / U
        return np.linalg.norm(diff, 'fro')
    else:
        diff = (V - pt.dot(W, H)) / U
        return pt.sqrt(pt.sum(diff ** 2))

def standardize_VU(V, U):
    V_mean = np.mean(V)
    V_std = np.std(V)
    V_scaled = (V - V_mean) / V_std
    U_scaled = U / V_std
    return V_scaled, U_scaled, V_mean, V_std

def rescale_W(W, V_std):
    return W * V_std

def find_elbow(penalty_weights):
    """
    Finds the elbow point in the penalty weights curve.
    penalty_weights: dict {k: penalty}
    Returns: optimal_k (int)
    """
    import numpy as np

    k_list = np.array(sorted(penalty_weights.keys()))
    penalties = np.array([penalty_weights[k] for k in k_list])

    # Line from first to last point
    x1, y1 = k_list[0], penalties[0]
    x2, y2 = k_list[-1], penalties[-1]

    # Compute distances from each point to the line
    numerator = np.abs((y2 - y1) * k_list - (x2 - x1) * penalties + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator

    # Elbow is at the max distance
    elbow_idx = np.argmax(distances)
    optimal_k = k_list[elbow_idx]
    return optimal_k

def run_model(V, U, min_factors:int=2, max_factors:int=10):
    W_k = {}
    H_k = {}
    q_k = {}
    for k in range(min_factors, max_factors + 1):
        batch_sa = BatchSA(V=V, U=U, factors=k, models=20, method="ls-nmf",
                           seed=int(rng.integers(low=0, high=1e8)), max_iter=20000,
                           converge_delta=0.1, converge_n=25, verbose=False, )
        _ = batch_sa.train()
        W_k[k] = batch_sa.results[batch_sa.best_model].W
        H_k[k] = batch_sa.results[batch_sa.best_model].H
        q_k[k] = batch_sa.results[batch_sa.best_model].Qtrue
    k_values = list(range(min_factors, max_factors + 1))

    k_list = sorted(q_k.keys())
    qtrue_min = np.min(list(q_k.values()))
    qtrue_max = np.max(list(q_k.values()))
    alpha = int(len(k_list)+min_factors) * (0.5 / (qtrue_min - qtrue_max + 1e-10))
    print(f"qtrue_min: {qtrue_min}, qtrue_max: {qtrue_max}, alpha: {alpha}")
    penalty_weights = {}
    penalty_weights[k_list[0]] = 0  # min_factors penalty is 0
    for i in range(1, len(k_list)):
        k = k_list[i]
        k_prev = k_list[i - 1]
        diff = q_k[k] - q_k[k_prev]
        penalty_weights[k] = (qtrue_max-qtrue_min) / abs(diff)
        print(f"K: {k}, Penalty weights: {penalty_weights[k]}")

    optimal_k = find_elbow(penalty_weights)
    # return optimal_k, W_k[optimal_k], H_k[optimal_k]

    W_padded = []
    H_padded = []
    for k in k_values:
        W = W_k[k]
        H = H_k[k]
        H_pad = np.zeros((max_factors, n_features))
        W_pad = np.zeros((n_samples, max_factors))
        W_pad[:, :k] = W
        H_pad[:k, :] = H
        W_padded.append(W_pad)
        H_padded.append(H_pad)
    W_stack = np.stack(W_padded)
    H_stack = np.stack(H_padded)

    with pm.Model() as model:
        prior_probs = np.exp(-alpha * (np.arange(min_factors, max_factors + 1) - min_factors))
        prior_probs /= pt.sum(prior_probs)
        k_values = np.arange(min_factors, max_factors + 1)

        # Compute penalized negative log-likelihood for each k
        penalized_nll = []
        for i, k in enumerate(k_values):
            W = W_stack[i]
            H = H_stack[i]
            mask = np.arange(max_factors) < k
            W_masked = W * mask[None, :]
            H_masked = H * mask[:, None]
            V_hat = pt.dot(W_masked, H_masked)
            # Negative log-likelihood (Gaussian)
            nll = 0.5 * pt.sum(((V - V_hat) / U) ** 2)
            # Penalty for complexity
            penalty = 2**penalty_weights[k]
            penalized_nll.append(nll+penalty)
            print(f"k={k}, nll={nll.eval() if hasattr(nll, 'eval') else nll}, penalty={penalty}")
        penalized_nll = pt.stack(penalized_nll)

        # Convert penalized losses to probabilities (lower loss = higher prob)
        logp = -penalized_nll
        logp = logp - pt.max(logp)  # for numerical stability
        probs = pt.exp(logp) * prior_probs
        probs /= pt.sum(probs)

        k_choice = pm.Categorical('k_choice', p=probs)

        trace = pm.sample(draws=4000, tune=2000, chains=6, cores=6, return_inferencedata=True)

    k_choice_posterior = trace.posterior['k_choice'].values.flatten()

    # Determine the most probable factor count
    optimal_k_index = np.argmax(np.bincount(k_choice_posterior))
    optimal_k = list(W_k.keys())[optimal_k_index]
    print(f"Optimal factor count: {optimal_k}")
    optimal_W = W_k[optimal_k]
    optimal_H = H_k[optimal_k]
    # W_rescaled = rescale_W(optimal_W, V_std)

    # ArviZ trace and posterior plots for k_choice
    az.plot_trace(trace, var_names=["k_choice"])
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    plt.suptitle("Trace plot for k_choice (factor count)")
    plt.show()

    az.plot_posterior(trace, var_names=["k_choice"])
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    plt.suptitle("Posterior distribution for k_choice (factor count)")
    plt.show()

    return optimal_k, optimal_W, optimal_H


if __name__ == "__main__":

    n_trials = 20  # Number of synthetic datasets to test
    min_factors = 4
    max_factors = 6
    max_iter = 20000
    results_file = "elbow_accuracy_results.json"
    results = {}
    total_correct = 0

    # Load previous results if file exists
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
            total_correct = results.get("total_correct", 0)

    pbar = tqdm(total=n_trials, desc="Running trials. Accuracy: NA, True K: NA, Est K: NA", unit="trial")
    for trial in range(n_trials):
        rng = np.random.default_rng(seed=trial)
        n_factors = int(rng.integers(low=min_factors, high=max_factors))
        n_samples = int(rng.integers(low=500, high=5000))
        n_features = int(rng.integers(low=20, high=50))

        simulator = Simulator(
            seed=int(rng.integers(low=0, high=1e10)),
            factors_n=n_factors,
            features_n=n_features,
            samples_n=n_samples,
            verbose=False,
        )
        syn_input_df, syn_uncertainty_df = simulator.get_data()
        data_handler = DataHandler.load_dataframe(input_df=syn_input_df, uncertainty_df=syn_uncertainty_df)
        V, U = data_handler.get_data()

        optimal_k, optimal_W, optimal_H = run_model(V, U, min_factors=min_factors, max_factors=max_factors+2)
        correct = int(optimal_k == n_factors)
        total_correct += correct

        # Track per-factor accuracy
        factor_str = str(n_factors)
        if factor_str not in results:
            results[factor_str] = {"correct": 0, "total": 0}
        results[factor_str]["correct"] += correct
        results[factor_str]["total"] += 1

        # Update overall accuracy
        results["total_correct"] = total_correct
        results["total_trials"] = trial + 1
        results["overall_accuracy"] = total_correct / (trial + 1)

        # Write results to file after each iteration
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        pbar.set_description(f"Running trials. Accuracy: {results['overall_accuracy']:.2f} True K: {n_factors}, Est K: {optimal_k}")
        pbar.update(1)
