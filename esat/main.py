from esat.data.datahandler import DataHandler
from esat.model.batch_sa import BatchSA
from esat_eval.simulator import Simulator
from memory_profiler import profile


def generate_data(
        seed: int = 42,
        syn_factors: int = 6,
        syn_features: int = 40,
        syn_samples: int = 100,
):
    seed = seed
    syn_factors = syn_factors  # Number of factors in the synthetic dataset
    syn_features = syn_features  # Number of features in the synthetic dataset
    syn_samples = syn_samples  # Number of samples in the synthetic dataset
    outliers = True  # Add outliers to the dataset
    outlier_p = 0.10  # Decimal percent of outliers in the dataset
    outlier_mag = 1.25  # Magnitude of outliers
    contribution_max = 2  # Maximum value of the contribution matrix (W) (Randomly sampled from a uniform distribution)
    noise_mean_min = 0.03  # Min value for the mean of noise added to the synthetic dataset, used to randomly determine the mean decimal percentage of the noise for each feature.
    noise_mean_max = 0.05  # Max value for the mean of noise added to the synthetic dataset, used to randomly determine the mean decimal percentage of the noise for each feature.
    noise_scale = 0.02  # Scale of the noise added to the synthetic dataset
    uncertainty_mean_min = 0.04  # Min value for the mean uncertainty of a data feature, used to randomly determine the mean decimal percentage for each feature in the uncertainty dataset.
    uncertainty_mean_max = 0.06  # Max value for the mean uncertainty of a data feature, used to randomly determine the mean decimal percentage for each feature in the uncertainty dataset.
    uncertainty_scale = 0.01  # Scale of the uncertainty matrix

    # Initialize the simulator with the above parameters
    simulator = Simulator(seed=seed,
                          factors_n=syn_factors,
                          features_n=syn_features,
                          samples_n=syn_samples,
                          outliers=outliers,
                          outlier_p=outlier_p,
                          outlier_mag=outlier_mag,
                          contribution_max=contribution_max,
                          noise_mean_min=noise_mean_min,
                          noise_mean_max=noise_mean_max,
                          noise_scale=noise_scale,
                          uncertainty_mean_min=uncertainty_mean_min,
                          uncertainty_mean_max=uncertainty_mean_max,
                          uncertainty_scale=uncertainty_scale
                         )

    # simulator.update_contribution(factor_i=0, curve_type="logistic", scale=0.1, frequency=0.5)
    # simulator.update_contribution(factor_i=1, curve_type="periodic", minimum=0.0, maximum=1.0, frequency=0.5, scale=0.1)
    # simulator.update_contribution(factor_i=2, curve_type="increasing", minimum=0.0, maximum=1.0, scale=0.1)
    # simulator.update_contribution(factor_i=3, curve_type="decreasing", minimum=0.0, maximum=1.0, scale=0.1)
    syn_input_df, syn_uncertainty_df = simulator.get_data()

    return syn_input_df, syn_uncertainty_df

@profile
def execute(batch_model):
    _ = batch_model.train()


def workflow(V, U, factors, models):
    method = "ls-nmf"  # "ls-nmf", "ws-nmf"
    init_method = "col_means"  # default is column means "col_means", "kmeans", "cmeans"
    seed = 42  # random seed for initialization
    max_iterations = 20000  # the maximum number of iterations for fitting a model
    converge_delta = 0.1  # convergence criteria for the change in loss, Q
    converge_n = 25  # convergence criteria for the number of steps where the loss changes by less than converge_delta
    parallel = True  # execute the model training in parallel, multiple models at the same time

    sa_models = BatchSA(V=V, U=U, factors=factors, models=models, method=method, seed=seed, max_iter=max_iterations,
                        init_method=init_method,
                        converge_delta=converge_delta, converge_n=converge_n,
                        parallel=parallel,
                        verbose=True
                        )
    execute(sa_models)


def run_all():
    input_df, uncertainty_df = generate_data(seed=42, syn_factors=7, syn_samples=1000, syn_features=34)
    # input_df, uncertainty_df = generate_data(seed=42, syn_factors=7, syn_samples=23446, syn_features=34)

    data_handler = DataHandler.load_dataframe(input_df=input_df, uncertainty_df=uncertainty_df)
    V, U = data_handler.get_data()
    workflow(V=V, U=U, factors=7, models=20)

if __name__ == '__main__':
    input_df, uncertainty_df = generate_data(seed=42, syn_factors=7, syn_samples=2000, syn_features=100)
    # input_df, uncertainty_df = generate_data(seed=42, syn_factors=7, syn_samples=23446, syn_features=34)
    # input_df, uncertainty_df = generate_data(seed=42, syn_factors=10, syn_samples=500000, syn_features=100)
    # input_df, uncertainty_df = generate_data(seed=42, syn_factors=10, syn_samples=1000000, syn_features=100)

    data_handler = DataHandler.load_dataframe(input_df=input_df, uncertainty_df=uncertainty_df)
    V, U = data_handler.get_data()
    workflow(V=V, U=U, factors=7, models=10)