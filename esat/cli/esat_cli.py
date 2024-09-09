import importlib.metadata
import sys
import os
import typer

import click
import json
import configparser
import logging
from importlib import metadata
from esat.data.datahandler import DataHandler
from esat.model.batch_sa import BatchSA
from esat.data.analysis import ModelAnalysis, BatchAnalysis
from esat.error.displacement import Displacement
from esat.error.bootstrap import Bootstrap
from esat.error.bs_disp import BSDISP
from esat.rotational.constrained import ConstrainedModel
from esat.configs import run_config, sim_config, error_config, constrained_config
from esat.estimator import FactorEstimator
try:
    from esat_eval.simulator import Simulator
except ModuleNotFoundError as e:
    from eval.simulator import Simulator


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    VERSION = metadata.version("esat")
except importlib.metadata.PackageNotFoundError as ex:
    logger.warning("ESAT package must be installed to determine version number")
    VERSION = "NA"


def get_dh(input_path, uncertainty_path, index_col):
    dh = DataHandler(input_path=input_path, uncertainty_path=uncertainty_path, index_col=index_col)
    return dh


def get_sim(sim_path, sim_parameters, sim_contributions):
    if "esat_simulator.pkl" in os.listdir(sim_path):
        sim = Simulator.load(os.path.join(sim_path, "esat_simulator.pkl"))
    else:
        sim = Simulator(**sim_parameters)
        for f_i, i_params in sim_contributions.items():
            c_params = json.loads(i_params)
            sim.update_contribution(factor_i=int(f_i), **c_params)
        _v, _u = sim.get_data()
        sim.save(output_directory=sim_path)

    return sim


def get_config(project_directory, error=False, constrained=False, sim=False):
    if error:
        config_file = os.path.join(project_directory, "error_config.toml")
    elif constrained:
        config_file = os.path.join(project_directory, "constrained_config.toml")
    elif sim:
        config_file = os.path.join(project_directory, "sim_config.toml")
    else:
        config_file = os.path.join(project_directory, "run_config.toml")
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def get_batchsa(project_directory):
    config = get_config(project_directory=project_directory)
    batch_pkl = os.path.join(config["project"]["directory"], "output", f"{config['project']['name']}.pkl")
    batch_sa = BatchSA.load(file_path=batch_pkl)
    return batch_sa


def get_model_analysis(project_directory, selected_model):
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    sa_models = get_batchsa(project_directory=project_directory)
    selected_i = sa_models.best_model if selected_model == -1 else selected_model
    i_model = sa_models.results[selected_i]
    ma = ModelAnalysis(datahandler=dh, model=i_model, selected_model=selected_i)
    return ma, selected_i


def get_error_model(project_directory, disp: bool = False, bs: bool = False, bsdisp: bool = False):
    config = get_config(project_directory=project_directory)
    error_dir = os.path.join(project_directory, "error")
    if disp:
        file_name = f"disp-{config['project']['name']}.pkl"
        return Displacement.load(os.path.join(error_dir, file_name))
    elif bs:
        file_name = f"bs-{config['project']['name']}.pkl"
        return Bootstrap.load(os.path.join(error_dir, file_name))
    elif bsdisp:
        file_name = f"bsdisp-{config['project']['name']}.pkl"
        return BSDISP.load(os.path.join(error_dir, file_name))
    else:
        return None


def get_constrained_model(project_directory):
    config = get_config(project_directory=project_directory)
    constrained_dir = os.path.join(project_directory, "constrained")
    file_name = f"constrained_model-{config['project']['name']}.pkl"
    return ConstrainedModel.load(os.path.join(constrained_dir, file_name))


@click.group()
@click.version_option(version=VERSION)
def esat_cli():
    """
    \b
    The EPA's Environmental Source Apportionment Toolkit (ESAT) CLI provides access to the ESAT workflow available in
    the Jupyter notebooks. The workflow sequence is as follows:
    \b
    1) setup : specify where you want your project directory for all solution outputs.
    2) analysis-input : (optional) review your input/uncertainty data with metric tables and plots.
    3) run : executes a batch source apportionment (SA) run using the values in the setup configuration file.
    4) analysis-solution : (optional) review the solutions of the batch SA run with metric tables and plots.
    5) setup-error : create an error estimation methods configuration file in the project directory.
    6) run-error: executes one or more of the error estimation methods using the error configuration.
    7) error-analysis : (optional) review the results of the error estimation methods.
    8) setup-constrained : create a constrained model run configuration file in the project directory.
    9) analysis-constrained : (optional) review the results of the constrained model run.

    An error estimation run can occur before or after a constrained model run, though the solution of a constrained
    model run can be used in an error estimation run by adding the path to the constrained model run configuration file
    in the error method configuration. If left empty, the error estimation run will default to the base solution from
    run.

    The ESAT simulator and synthetic datasets can be used by running the 'setup-sim' command in place of 'setup'. To
    generate the synthetic dataset for analysis run 'generate-sim', if this is not executed prior to 'run' the command
    will be executed in the run command sequence. The 'run' command will use the generated synthetic data as it would
    real data. The model outputs can then be compared to the synthetic profiles with the 'sim-analysis' command.
    All other commands are available on the simulated data.

    """
    pass


@esat_cli.command()
@click.argument("project_directory", type=click.Path())
def setup(project_directory):
    """
    Create the configuration file for a batch source apportionment run in the provided directory.

    Parameters

    project_directory : The project directory where all configuration output files are saved.

    """
    try:
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
    except FileNotFoundError:
        logger.error("Unable to create workflow directory, make sure the path is correct.")
        return
    logger.info(f"Creating new ESAT project")
    new_config = run_config
    new_config['project']['directory'] = project_directory
    new_config_file = os.path.join(project_directory, "run_config.toml")
    with open(new_config_file, 'w') as configfile:
        new_config.write(configfile)
    logger.info(f"New run configuration file created. File path: {new_config_file}")


@esat_cli.group()
def simulator():
    """
    The collection of commands for managing ESAT simulator instances.
    """
    pass


@simulator.command(name='setup')
@click.argument("project_directory", type=click.Path())
def setup_sim(project_directory):
    """
    Create the configuration file for a batch source apportionment simulated run in the provided directory.

    Parameters

    project_directory : The project directory where all configuration output files are saved.

    """
    try:
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
    except FileNotFoundError:
        logger.error("Unable to create workflow directory, make sure the path is correct.")
        return
    logger.info(f"Creating new ESAT simulator project")
    new_sim_config = sim_config
    new_sim_config['project']['directory'] = project_directory
    new_sim_config_file = os.path.join(project_directory, "sim_config.toml")
    with open(new_sim_config_file, 'w') as configfile:
        new_sim_config.write(configfile)
    logger.info(f"New simulator configuration file created. File path: {new_sim_config_file}")
    new_config = run_config
    new_config['project']['directory'] = project_directory
    new_config['data']['input_path'] = new_sim_config['data']['input_path']
    new_config['data']['uncertainty_path'] = new_sim_config['data']['uncertainty_path']
    new_config_file = os.path.join(project_directory, "run_config.toml")
    with open(new_config_file, 'w') as configfile:
        new_config.write(configfile)
    logger.info(f"New run configuration file created. File path: {new_config_file}")


@simulator.command(name="generate")
@click.argument("project_directory", type=click.Path())
def generate_sim(project_directory):
    """
    Generate the synthetic data and uncertainty datasets as defined in the sim_config.toml. Only required for data
    analysis functions, run will check for a simulated config if no file is found in the project_directory and generate
    if required.

    Parameters

    project_directory : The project directory where all configuration output files are saved.
    """
    try:
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
    except FileNotFoundError:
        logger.error("Unable to create workflow directory, make sure the path is correct.")
        return
    config = get_config(project_directory=project_directory, sim=True)
    logger.info("Generating synthetic data")
    sim = get_sim(sim_path=project_directory,
                  sim_parameters=config['parameters'],
                  sim_contributions=config["contributions"]
                  )
    logger.info("ESAT Simulator setup complete")


@simulator.command(name="compare")
@click.argument("project_directory", type=click.Path())
def compare_sim(project_directory):
    """
    Compares the models generated from the synthetic data to show how similar the modelled profiles are to the known
    synthetic profiles.

    Parameters

    project_directory : The project directory where all configuration output files are saved.
    """
    try:
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
    except FileNotFoundError:
        logger.error("Unable to create workflow directory, make sure the path is correct.")
        return
    config = get_config(project_directory=project_directory, sim=True)
    logger.info("Generating synthetic data")
    sim = get_sim(sim_path=project_directory,
                  sim_parameters=config['parameters'],
                  sim_contributions=config["contributions"]
                  )
    batch_sa = get_batchsa(project_directory=project_directory)
    sim.compare(batch_sa=batch_sa)
    sim.save(output_directory=project_directory)


@simulator.command(name="plot")
@click.argument("project_directory", type=click.Path())
def plot_sim(project_directory):
    """
    Plots the results of the simulator output comparison to the synthetic profiles.

    Parameters

    project_directory : The project directory where all configuration output files are saved.
    """
    try:
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
    except FileNotFoundError:
        logger.error("Unable to create workflow directory, make sure the path is correct.")
        return
    config = get_config(project_directory=project_directory, sim=True)
    logger.info("Generating synthetic data")
    sim = get_sim(sim_path=project_directory,
                  sim_parameters=config['parameters'],
                  sim_contributions=config["contributions"]
                  )
    sim.plot_comparison()


@esat_cli.group()
def analysis_batch():
    """
    The collection of commands for batch model analysis.
    """
    pass


@analysis_batch.command()
@click.argument("project_directory", type=click.Path())
def plot_loss(project_directory):
    """
    Plots the loss value Q(true) over the training iterations for the batch solution.

    Parameters
    ----------
    project_directory : The project directory where all configuration output files are saved.
    """
    try:
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
    except FileNotFoundError:
        logger.error("Unable to create workflow directory, make sure the path is correct.")
        return
    batch_sa = get_batchsa(project_directory=project_directory)
    batch_analysis = BatchAnalysis(batch_sa=batch_sa)
    batch_analysis.plot_loss()


@analysis_batch.command()
@click.argument("project_directory", type=click.Path())
def plot_distribution(project_directory):
    """
    Plots the loss value distribution Q(true) and Q(robust) for the batch solution.

    Parameters
    ----------
    project_directory : The project directory where all configuration output files are saved.
    """
    try:
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
    except FileNotFoundError:
        logger.error("Unable to create workflow directory, make sure the path is correct.")
        return
    batch_sa = get_batchsa(project_directory=project_directory)
    batch_analysis = BatchAnalysis(batch_sa=batch_sa)
    batch_analysis.plot_loss_distribution()


@esat_cli.group()
def analysis_input():
    """
    The collection of commands for analyzing the input/uncertainty datasets.
    """
    pass


@analysis_input.command(name="metrics")
@click.argument("project_directory", type=click.Path(exists=True))
def metrics(project_directory):
    """
    Display the input metrics for the input/uncertainty data specified in the configuration file.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    logger.info("Input Dataset Metrics")
    dh = get_dh(**config["data"])
    logger.info(dh.metrics)


@analysis_input.command(name="plot-feature")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-i', "--feature_idx", type=int, default=0, help="The index of the input/uncertainty data feature, "
                                                               "column in the dataset.", show_default=True)
def plot_input_uncertainty(project_directory, feature_idx):
    """
    Display the Concentration/Uncertainty scatter plot for the specified feature.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    dh.plot_data_uncertainty(feature_idx=feature_idx)


@analysis_input.command(name="plot-data")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-x', "--feature_idx1", type=int, default=0, help="The index of the first (x-axis) input data feature, "
                                                                "column in the dataset.", show_default=True)
@click.option('-y', "--feature_idx2", type=int, default=1, help="The index of the second (y-axis) input data feature, "
                                                                "column in the dataset.", show_default=True)
def plot_feature_data(project_directory, feature_idx1, feature_idx2):
    """
    Display the feature vs feature scatter plot.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    dh.plot_feature_data(x_idx=feature_idx1, y_idx=feature_idx2)


@analysis_input.command(name="plot-timeseries")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-i', "--feature_idx", type=int, default=0, help="The indices of the input data feature, "
                                                               "columns in the dataset.", show_default=True)
def plot_feature_timeseries(project_directory, feature_idx):
    """
    Display the feature(s) timeseries.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    dh.plot_feature_timeseries(feature_selection=list(feature_idx))


@esat_cli.command()
@click.argument("project_directory", type=click.Path(exists=True))
def estimate(project_directory):
    """
    Run a factor estimation using the provided configuration file.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    if "sim_config.toml" in os.listdir(project_directory):
        _sim_config = get_config(project_directory=project_directory, sim=True)
        sim = get_sim(sim_path=project_directory,
                      sim_parameters=_sim_config['parameters'],
                      sim_contributions=_sim_config['contributions'])
        data_df, uncertainty_df = sim.get_data()
        dh = DataHandler.load_dataframe(input_df=data_df, uncertainty_df=uncertainty_df)
    else:
        dh = DataHandler(**config["data"])
    V, U = dh.get_data()
    factor_estimator = FactorEstimator(V=V, U=U)
    estimator_results = factor_estimator.run(
        samples=int(config["estimator"]["samples"]),
        min_factors=int(config["estimator"]["min_k"]),
        max_factors=int(config["estimator"]["max_k"])
    )
    logger.info(f"{estimator_results}")


@esat_cli.command()
@click.argument("project_directory", type=click.Path(exists=True))
def run(project_directory):
    """
    Run a batch source apportionment run using the provided configuration file.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    if "sim_config.toml" in os.listdir(project_directory):
        _sim_config = get_config(project_directory=project_directory, sim=True)
        sim = get_sim(sim_path=project_directory,
                      sim_parameters=_sim_config['parameters'],
                      sim_contributions=_sim_config['contributions'])
        data_df, uncertainty_df = sim.get_data()
        dh = DataHandler.load_dataframe(input_df=data_df, uncertainty_df=uncertainty_df)
        logger.info(f"Executing ESAT Simulation")
    else:
        dh = DataHandler(**config["data"])
    V, U = dh.get_data()
    sa_models = BatchSA(V=V, U=U, **config["parameters"])
    sa_models.details()
    _ = sa_models.train()
    output_path = os.path.join(config["project"]["directory"], "output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    sa_models.save(batch_name=config["project"]["name"], output_directory=output_path, pickle_batch=True)
    sa_models.save(batch_name=config["project"]["name"], output_directory=output_path, pickle_batch=False,
                   pickle_model=False, header=dh.features)


@esat_cli.group()
def analysis_solution():
    """
    The collection of commands for analyzing an SA project base solution.
    """
    pass


@analysis_solution.command(name="plot-residuals")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
@click.option('-i', "--feature_idx", default=0, type=int, help="The index of the feature to plot.", show_default=True)
def solution_residual_histogram(project_directory, selected_model, feature_idx):
    """
    Plot the residual histogram for a specified feature index (feature_idx) and solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Residual Histogram for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_residual_histogram(feature_idx=feature_idx)


@analysis_solution.command(name="statistics")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
def solution_statistics(project_directory, selected_model):
    """
    Display the solution statistics for a specified solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Statistics. Model selected: {selected_i}")
    ma.calculate_statistics()
    logger.info(ma.statistics)


@analysis_solution.command(name="plot-estimated")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
@click.option('-i', "--feature_idx", default=0, type=int, help="The index of the feature to plot.", show_default=True)
def solution_estimated_observed(project_directory, selected_model, feature_idx):
    """
    Plot the estimated vs observed for a specified feature index (feature_idx) and solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Estimated/Observed for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_estimated_observed(feature_idx=feature_idx)


@analysis_solution.command(name="plot-timeseries")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
@click.option('-i', "--feature_idx", default=0, type=int, help="The index of the feature to plot.", show_default=True)
def solution_estimated_timeseries(project_directory, selected_model, feature_idx):
    """
    Plot the estimated timeseries for a specified feature index (feature_idx) and solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Estimated Timeseries for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_estimated_timeseries(feature_idx=feature_idx)


@analysis_solution.command(name="plot-profile")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
@click.option('-i', "--factor_idx", default=0, type=int, help="The index of the factor profile.", show_default=True)
def solution_factor_profile(project_directory, selected_model, factor_idx):
    """
    Plot the factor profile for a specified factor index (factor_idx) and solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Profile for model: {selected_i}, factor index: {factor_idx}")
    ma.plot_factor_profile(factor_idx=factor_idx)


@analysis_solution.command(name="plot-fingerprints")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
def solution_factor_fingerprints(project_directory, selected_model):
    """
    Plot the factor fingerprints for a specified solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Fingerprints for model: {selected_i}")
    ma.plot_factor_fingerprints()


@analysis_solution.command(name="plot-gspace")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
@click.option('-x', "--factor_idx1", default=0, type=int, help="The index of the x-axis factor profile.",
              show_default=True)
@click.option('-y', "--factor_idx2", default=1, type=int, help="The index of the y-axis factor profile.",
              show_default=True)
def solution_g_space(project_directory, selected_model, factor_idx1, factor_idx2):
    """
    Plot two factor g-space graph for a specified solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution G-Space Plot for model: {selected_i}, factor 1: {factor_idx1}, factor 2: {factor_idx2}")
    ma.plot_g_space(factor_1=factor_idx1, factor_2=factor_idx2)


@analysis_solution.command(name="plot-contributions")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
@click.option('-i', "--feature_idx", default=0, type=int, help="The index of the feature to plot.", show_default=True)
def solution_factor_contributions(project_directory, selected_model, feature_idx):
    """
    Plot factor contributions for a specified feature index (feature_idx) and solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Contributions for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_factor_contributions(feature_idx=feature_idx)


@analysis_solution.command(name="plot-composition")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
def solution_factor_composition(project_directory, selected_model):
    """
    Plot factor composition for a specified solution model (selected_model).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Composition for model: {selected_i}")
    ma.plot_factor_composition()


@analysis_solution.command(name="plot-surface")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-m', "--selected_model", default=-1, type=int, help="The index of the model from the solution. "
                                                                   "Default: -1, the best performing model.",
              show_default=True)
@click.option('-i', "--factor_idx", default=0, type=int, help="The index of the factor profile.", show_default=True)
@click.option('-f', "--feature_idx", type=int, multiple=True, help="The index/indices of the factor feature to "
                                                                   "include. Default: -1, includes all features which "
                                                                   "contribution to the factor_idx. Multiple indices"
                                                                   " can be provided.")
def solution_factor_surface(project_directory, selected_model, factor_idx, feature_idx):
    """
    Plot factor surface for a specified factor index (factor_idx), set of features (feature_idx), and solution model
    (selected_model). Multiple feature indices can be provided as '-f 0 -f 1 -f 2'.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    feature_idx = feature_idx if len(feature_idx) > 0 else -1
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    factor_idx = factor_idx if factor_idx != -1 else None
    feature_idx = feature_idx if feature_idx != -1 else None
    logger.info(f"ESAT Solution Factor Surface for model: {selected_i}, factor index: {factor_idx}, "
                f"feature index: {feature_idx}")
    ma.plot_factor_surface(feature_idx=feature_idx, factor_idx=factor_idx)


@esat_cli.command()
@click.argument("project_directory", type=click.Path(exists=True))
def setup_error(project_directory):
    """
    Create an error estimation configuration file for the specified project.

    Parameters

    project_directory : The project directory where all output files are saved.

    """
    new_config = error_config
    new_config["project"]["project_config"] = os.path.join(project_directory, "run_config.toml")
    new_config_file = os.path.join(project_directory, "error_config.toml")
    with open(new_config_file, 'w') as configfile:
        new_config.write(configfile)
    logger.info(f"New error configuration file created. File path: {new_config_file}")


@esat_cli.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--disp", is_flag=True, default=False,
              help="Run the displacement (DISP) error estimation method.", show_default=True)
@click.option("--bs", is_flag=True, prompt=False, default=False, show_default=True,
              help="Run the bootstrap (BS) error estimation method.")
@click.option("--bsdisp", is_flag=True, prompt=False, show_default=True, default=False,
              help="Run the bootstrap-displacement (BSDISP) error estimation method.")
def run_error(project_directory, disp, bs, bsdisp):
    """
    Run one or more of the error estimation methods using the specified error configuration file.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    if not any((disp, bs, bsdisp)):
        logger.info("One or more error estimation method must be selected.")
        logger.info(f"DISP: {disp}, BS: {bs}, BS-DISP: {bsdisp}")
        return
    config = get_config(project_directory=project_directory)
    e_config = get_config(project_directory=project_directory, error=True)

    dh = DataHandler(**config["data"])
    batch_pkl = os.path.join(project_directory, "output", f"{config['project']['name']}.pkl")
    sa_models = BatchSA.load(file_path=batch_pkl)
    error_dir = os.path.join(project_directory, "error")
    if not os.path.exists(error_dir):
        os.mkdir(error_dir)

    if e_config["project"]["constrained_config"] != "":
        constrained_pkl = os.path.join(project_directory, "constrained",
                                       f"constrained_{config['project']['name']}.pkl")
        c_solution = ConstrainedModel.load(file_path=constrained_pkl)
        logger.info("Using existing constrained model.")
        selected_i = -1
        selected_model = c_solution.constrained_model
    else:
        selected_i = int(e_config["project"]["selected_model"])
        selected_i = sa_models.best_model if selected_i == -1 else selected_i
        selected_model = sa_models.results[selected_i]
    logger.info(f"Running error estimation on model {selected_i}")
    if disp:
        features = json.loads(e_config["disp"]["features"])
        features = [int(f) for f in features] if len(features) >= 1 else None
        features_label = features if features is not None else "all"
        logger.info(f"Running DISP on model {selected_i} for features: {features_label}")
        disp = Displacement(
            sa=selected_model,
            feature_labels=dh.features,
            model_selected=selected_i,
            features=features
        )
        disp.run()
        disp.save(disp_name=f"disp-{config['project']['name']}", output_directory=error_dir, pickle_result=True)
        disp.save(disp_name=f"disp-{config['project']['name']}", output_directory=error_dir, pickle_result=False)
        logger.info("DISP completed.")
    if bs:
        block_size = int(e_config["bs"]["block_size"])
        block_size = dh.optimal_block if block_size == -1 else block_size
        seed = int(e_config["bs"]["seed"])
        seed = selected_model.seed if seed == -1 else seed
        logger.info(f"Running BS on model {selected_i}, block size: {block_size}, seed: {seed}")
        bs = Bootstrap(
            sa=selected_model,
            feature_labels=dh.features,
            model_selected=selected_i,
            bootstrap_n=int(e_config["bs"]["bootstrap_n"]),
            block_size=block_size,
            threshold=float(e_config["bs"]["threshold"]),
            seed=seed
        )
        bs.run()
        bs.save(bs_name=f"bs-{config['project']['name']}", output_directory=error_dir, pickle_result=True)
        bs.save(bs_name=f"bs-{config['project']['name']}", output_directory=error_dir, pickle_result=False)
        logger.info("BS Completed.")
    if bsdisp:
        block_size = int(e_config["bsdisp"]["block_size"])
        block_size = dh.optimal_block if block_size == -1 else block_size
        seed = int(e_config["bsdisp"]["seed"])
        seed = selected_model.seed if seed == -1 else seed
        features = json.loads(e_config["disp"]["features"])
        features = [int(f) for f in features] if len(features) >= 1 else None
        features_label = features if features is not None else "all"
        logger.info(f"Running DISP on model {selected_i} for features: {features_label}")
        logger.info(f"Running BSDISP on model {selected_i}, features: {features_label}")
        _bs = None
        if e_config["bsdisp"]["bootstrap_output"] != "":
            bs_output_file = e_config["bsdisp"]["bootstrap_output"]
            if os.path.exists(bs_output_file):
                _bs = Bootstrap.load(bs_output_file)
                logger.info("Using existing BS solution.")
        elif bs is not None:
            _bs = bs
            logger.info("Using existing BS solution.")
        bsdisp = BSDISP(
            sa=selected_model,
            bootstrap=_bs,
            feature_labels=dh.features,
            model_selected=selected_i,
            bootstrap_n=int(e_config["bsdisp"]["bootstrap_n"]),
            block_size=block_size,
            threshold=float(e_config["bsdisp"]["threshold"]),
            seed=seed,
            features=features,
            max_search=int(e_config["bsdisp"]["max_search"]),
            threshold_dQ=float(e_config["bsdisp"]["threshold_dQ"])
        )
        parallel = config['parameters']['parallel'].lower() == "true"
        bsdisp.run(parallel=parallel)
        bsdisp.save(bsdisp_name=f"bsdisp-{config['project']['name']}", output_directory=error_dir, pickle_result=True)
        bsdisp.save(bsdisp_name=f"bsdisp-{config['project']['name']}", output_directory=error_dir, pickle_result=False)
        logger.info("BSDISP Completed")


@esat_cli.group()
def analysis_error():
    """
    The collection of commands for analyzing the results of error estimation methods.
    """
    pass


@analysis_error.command(name="summary")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--disp", is_flag=True, prompt_required=False, default=False, show_default=True,
              help="Summarize the displacement (DISP) error estimation results.")
@click.option("--bs", is_flag=True, prompt_required=False, default=False, show_default=True,
              help="Summarize the bootstrap (BS) error estimation results.")
@click.option("--bsdisp", is_flag=True, prompt_required=False, default=False, show_default=True,
              help="Summarize the bootstrap-displacement (BSDISP) error estimation results.")
def error_summary(project_directory, disp, bs, bsdisp):
    """
    Summarize the results of an error estimation method results.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    if not all((disp, bs, bsdisp)):
        logger.info("One error estimation method must be selected.")
        return
    error_solution = get_error_model(
        project_directory=project_directory,
        disp=disp,
        bs=bs,
        bsdisp=bsdisp
    )
    error_solution.summary()


@analysis_error.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--disp", is_flag=True, prompt_required=False, default=False, show_default=True,
              help="Summarize the displacement (DISP) error estimation results.")
@click.option("--bs", is_flag=True, prompt_required=False, default=False, show_default=True,
              help="Summarize the bootstrap (BS) error estimation results.")
@click.option("--bsdisp", is_flag=True, prompt_required=False, default=False, show_default=True,
              help="Summarize the bootstrap-displacement (BSDISP) error estimation results.")
@click.option("--factor_idx", default=0, type=int, help="The factor index to display results.")
def error_results(project_directory, disp, bs, bsdisp, factor_idx):
    """
    Show the results of an error estimation method. One method must be selected.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    if not all((disp, bs, bsdisp)):
        logger.info("One error estimation method must be selected.")
        return
    error_solution = get_error_model(
        project_directory=project_directory,
        disp=disp,
        bs=bs,
        bsdisp=bsdisp
    )
    error_solution.plot_results(factor=factor_idx)


@esat_cli.command()
@click.argument("project_directory", type=click.Path(exists=True))
def setup_constrained(project_directory):
    """
    Create a configuration file for executing a constrained model run on a specified project.

    Parameters

    project_directory : The project directory where all output files are saved.

    """
    new_config = constrained_config
    new_config["project"]["project_config"] = os.path.join(project_directory, "run_config.toml")
    new_config_file = os.path.join(project_directory, "constrained_config.toml")
    with open(new_config_file, 'w') as configfile:
        new_config.write(configfile)
    logger.info(f"New constrained configuration file created. File path: {new_config_file}")


@esat_cli.command()
@click.argument("project_directory", type=click.Path(exists=True))
def run_constrained(project_directory):
    """
    Run a constrained model with the specified configuration file.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    c_config = get_config(project_directory=project_directory, constrained=True)

    dh = DataHandler(**config["data"])

    batch_pkl = os.path.join(project_directory, "output", f"{config['project']['name']}.pkl")
    sa_models = BatchSA.load(file_path=batch_pkl)
    selected_i = c_config["parameters"]["selected_model"]
    selected_i = sa_models.best_model if int(selected_i) == -1 else int(selected_i)
    selected_model = sa_models.results[selected_i]

    constrained_dir = os.path.join(project_directory, "constrained")
    constrained_model = ConstrainedModel(base_model=selected_model,
                                         data_handler=dh,
                                         softness=float(c_config["parameters"]["softness"])
                                         )
    logger.info(f"Running Constrained Model run on model {selected_i}")
    for c_key in c_config["constraints"]:
        constraint = c_config["constraints"][c_key]
        c_dict = json.loads(constraint)
        if c_dict['index'] == [-1, -1]:
            logger.info(f"Excluding constraint: {c_key}. Constraints with index values of -1 are not included.")
        else:
            c_dict["index"] = (int(c_dict["index"][0]), int(c_dict["index"][1]))
            constrained_model.add_constraint(**c_dict)
    constrained_model.list_constraints()

    for e_key in c_config["expressions"]:
        expression = c_config["expressions"][e_key]
        if any(label in expression for label in ("factor:-1", "feature:-1", "sample:-1")):
            logger.info(f"Excluding expression: {e_key}. Expressions with index values of -1 are not included.")
        else:
            constrained_model.add_expression(expression)
    constrained_model.list_expressions()
    max_iter = int(c_config["parameters"]["max_iter"]) if int(c_config["parameters"]["max_iter"]) != -1 else sa_models.max_iter
    converge_delta = float(c_config["parameters"]["converge_delta"]) if float(c_config["parameters"]["converge_delta"]) > 0.0 else sa_models.converge_delta
    converge_n = int(c_config["parameters"]["converge_n"]) if int(c_config["parameters"]["converge_n"]) != -1 else sa_models.converge_n
    constrained_model.train(max_iterations=max_iter,
                            converge_delta=converge_delta,
                            converge_n=converge_n)
    if constrained_model.constrained_model is not None:
        constrained_model.display_results()
        if not os.path.exists(constrained_dir):
            os.mkdir(constrained_dir)
        constrained_model.save(model_name=config["project"]["name"], output_directory=constrained_dir, pickle_model=True)
        constrained_model.save(model_name=config["project"]["name"], output_directory=constrained_dir, pickle_model=False)
        logger.info("Constrained model run completed.")
    else:
        logger.error("Unable to complete constrained model run.")


@esat_cli.group()
def analysis_constrained():
    """
    The collection of commands for analyzing the constrained model results.
    """
    pass


@analysis_constrained.command(name="summary")
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_results(project_directory):
    """
    Display the results of a constrained model run.

    Parameters

    project_directory : The project directory containing .toml configuration files.


    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.display_results()


@analysis_constrained.command(name="plot-q")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-q', "--qtype", prompt_required=False, default='Aux', show_default=True,
              help="Plot the loss value for qtype: True, Robust, Aux")
def constrained_plot_q(project_directory, qtype):
    """
    Display the loss function plotted over all training iterations. Loss value options for Q include: True, Robust, and
    Aux (the loss value difference between the current solution values and the target values as defined from the
    constraints and expressions).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_Q(Qtype=qtype)


@analysis_constrained.command(name="eval-constraints")
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_evaluate_constraints(project_directory):
    """
    Display the evaluation table of the constrained model results, for each constraint.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.evaluate_constraints()


@analysis_constrained.command(name="eval-expressions")
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_evaluate_expressions(project_directory):
    """
    Display the evaluation table of the constrained model results, for each expression.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.evaluate_expressions()


@analysis_constrained.command(name="plot-contributions")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-i', "--factor_idx", default=0, type=int, show_default=True, help="The factor index to plot.")
def constrained_profile_contributions(project_directory, factor_idx):
    """
    Display a specified factor profile and contributions plot for the constrained model solution.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_profile_contributions(factor_idx=factor_idx)


@analysis_constrained.command(name="plot-fingerprints")
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_factor_fingerprints(project_directory):
    """
    Display the constrained solution factor fingerprints.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_factor_fingerprints()


@analysis_constrained.command(name="plot-gspace")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-x', "--factor_idx1", default=0, type=int, show_default=True, help="The factor index for the x-axis.")
@click.option('-y', "--factor_idx2", default=1, type=int, show_default=True, help="The factor index for the y-axis.")
@click.option("--show_base", is_flag=True, default=True, type=bool, show_default=True,
              help="Show the base solution values.")
@click.option("--show_delta", is_flag=True, default=True, type=bool, show_default=True,
              help="Show the change between the constrained solution values and  the base solution values")
def constrained_g_space(project_directory, factor_idx1, factor_idx2, show_base, show_delta):
    """
    Display a specified factor index (factor_idx1) vs factor index (factor_idx2) g-space plot.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_g_space(factor_idx1=factor_idx1, factor_idx2=factor_idx2, show_base=show_base, show_delta=show_delta)


@analysis_constrained.command(name="plot-contributions")
@click.argument("project_directory", type=click.Path(exists=True))
@click.option('-i', "--feature_idx", default=0, type=int, show_default=True, help="The feature index to display.")
def constrained_profile_contributions(project_directory, feature_idx):
    """
    Display the factor contributions for the specified feature (feature_idx).

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_factor_contributions(feature_idx=feature_idx)


if __name__ == "__main__":
    typer.run(esat_cli)
