import pytest
import numpy as np
from esat_eval.perturbation import Perturbation
from esat.model.sa import SA


@pytest.fixture
def setup_perturbation():
    V = np.random.rand(10, 5)
    U = np.random.rand(10, 5)
    factors = 3
    base_model = SA(V=V, U=U, factors=factors, seed=42, verbose=False, method="ls-nmf")
    base_model.initialize()
    base_model.train(max_iter=100, converge_delta=0.001, converge_n=10)

    perturbation = Perturbation(
        V=V,
        U=U,
        factors=factors,
        base_model=base_model,
        random_seed=42,
        perturb_percent=0.5,
        sigma=0.1,
        models=5,
        max_iterations=100,
        converge_n=10,
        converge_delta=0.001,
        threshold=0.1,
        compare_method="raae",
        verbose=False
    )
    return perturbation, V, U, factors


def run_perturbation(perturbation):
    """Helper function to execute the run method."""
    perturbation.run()


def test_initialization(setup_perturbation):
    perturbation, _, _, factors = setup_perturbation
    assert perturbation.factors == factors
    assert perturbation.method == "ls-nmf"
    assert perturbation.random_seed == 42
    assert perturbation.models == 5


def test_perturb(setup_perturbation):
    perturbation, _, U, _ = setup_perturbation
    perturbed_U, perturbed_multipliers = perturbation.perturb(random_seed=42)
    assert perturbed_U.shape == U.shape
    assert perturbed_multipliers.shape == U.shape
    assert np.all(perturbed_U > 0)  # Ensure no negative values


def test_run(setup_perturbation):
    perturbation, _, _, _ = setup_perturbation
    run_perturbation(perturbation)
    assert perturbation.perturbed_models is not None
    assert perturbation.perturbed_multipliers is not None
    assert len(perturbation.perturbed_models) == perturbation.models


def test_compare(setup_perturbation):
    perturbation, _, _, _ = setup_perturbation
    run_perturbation(perturbation)
    perturbation.compare(compare_method="raae")
    assert perturbation.comparison is not None
    assert perturbation.perturb_mapping is not None
    assert perturbation.perturb_correlations is not None


def test_plot_norm_perturb_contributions(setup_perturbation):
    perturbation, _, _, _ = setup_perturbation
    run_perturbation(perturbation)
    perturbation.compare()
    fig = perturbation.plot_norm_perturb_contributions(return_figure=True)
    assert fig is not None


def test_plot_average_source_contributions(setup_perturbation):
    perturbation, _, _, _ = setup_perturbation
    run_perturbation(perturbation)
    perturbation.compare()
    fig = perturbation.plot_average_source_contributions(return_figure=True)
    assert fig is not None


def test_plot_correlation_metrics(setup_perturbation):
    perturbation, _, _, _ = setup_perturbation
    run_perturbation(perturbation)
    perturbation.compare()
    fig = perturbation.plot_correlation_metrics(return_figure=True)
    assert fig is not None


def test_plot_perturbed_factor(setup_perturbation):
    perturbation, _, _, _ = setup_perturbation
    run_perturbation(perturbation)
    perturbation.compare()
    fig = perturbation.plot_perturbed_factor(factor_idx=0, return_figure=True)
    assert fig is not None