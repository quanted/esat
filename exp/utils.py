import warnings
from typing import Tuple
import numpy as np
warnings.filterwarnings('ignore')


from esat_eval.simulator import Simulator
from esat.data.datahandler import DataHandler


def generate_dataset(true_k=None, n_samples=None, n_features=None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, Simulator]:
    """Generate a single synthetic dataset with varied parameters"""
    # Vary the experimental conditions
    np.random.seed(seed)
    n_samples = np.random.randint(1000, 10000) if n_samples is None else n_samples
    n_features = np.random.randint(8, 40) if n_features is None else n_features
    true_factors = np.random.randint(3, 9) if true_k is None else true_k
    noise_min = np.random.uniform(0.1, 0.2)
    noise_max = np.random.uniform(0.2, 0.3)
    noise_scale = np.random.uniform(0.1, 0.2)
    outliers = np.random.uniform(0.1, 0.3)
    outlier_mag = np.random.uniform(1.5, 5.0)

    simulator = Simulator(seed=np.random.random_integers(low=0, high=1000),
                          factors_n=true_factors,
                          features_n=n_features,
                          samples_n=n_samples,
                          outliers=True,
                          outlier_p=outliers,
                          outlier_mag=outlier_mag,
                          contribution_max=2,
                          noise_mean_min=noise_min,
                          noise_mean_max=noise_max,
                          noise_scale=noise_scale,
                          uncertainty_mean_min=0.04,
                          uncertainty_mean_max=0.07,
                          uncertainty_scale=0.01,
                          verbose=False
                          )
    syn_input_df, syn_uncertainty_df = simulator.get_data()
    data_handler = DataHandler.load_dataframe(input_df=syn_input_df, uncertainty_df=syn_uncertainty_df)
    V, U = data_handler.get_data()
    return V, U, true_factors, simulator.syn_profiles, simulator.syn_contributions, simulator

def q_loss(V, U, W, H):
    _wh = np.matmul(W, H)
    residuals = np.subtract(V, _wh)
    residuals_u = np.divide(residuals, U)
    r2 = np.multiply(residuals_u, residuals_u)
    _q = np.sum(r2)
    return _q
