import sys
import os

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "..\\src")

import click
import json
import configparser
import logging
from src.data.datahandler import DataHandler
from src.model.batch_sa import BatchSA
from src.data.analysis import ModelAnalysis
from src.error.displacement import Displacement
from src.error.bootstrap import Bootstrap
from src.error.bs_disp import BSDISP
from src.rotational.constrained import ConstrainedModel
from src.configs import run_config, error_config, constrained_config

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dh(input_path, uncertainty_path, index_col):
    dh = DataHandler(input_path=input_path, uncertainty_path=uncertainty_path, index_col=index_col)
    return dh


def get_config(project_directory, error=False, constrained=False):
    if error:
        config_file = os.path.join(project_directory, "error_config.toml")
    elif constrained:
        config_file = os.path.join(project_directory, "constrained_config.toml")
    else:
        config_file = os.path.join(project_directory, "run_config.toml")
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def get_model_analysis(project_directory, selected_model):
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    batch_pkl = os.path.join(config["project"]["directory"], "output", f"{config['project']['name']}.pkl")
    sa_models = BatchSA.load(file_path=batch_pkl)
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
def esat():
    """
    \b
    The EPA's Environmental Source Apportionment Toolkit (ESAT) CLI provides access to the ESAT workflow available in the
    Jupyter notebooks. The workflow sequence is as follows:
    \b
    1) setup : specify where you want your project directory for all solution outputs.
    2) input-analysis : (optional) review your input/uncertainty data with metric tables and plots.
    3) run : executes a batch source apportionment (SA) run using the values in the setup configuration file.
    4) solution-analysis : (optional) review the solutions of the batch SA run with metric tables and plots.
    5) setup-error : create an error estimation methods configuration file in the project directory.
    6) run-error: executes one or more of the error estimation methods using the error configuration.
    7) error-analysis : (optional) review the results of the error estimation methods.
    8) setup-constrained : create a constrained model run configuration file in the project directory.
    9) constrained-analysis : (optional) review the results of the constrained model run.

    An error estimation run can occur before or after a constrained model run, though the solution of a constrained
    model run can be used in an error estimation run by adding the path to the constrained model run configuration file
    in the error method configuration. If left empty, the error estimation run will default to the base solution from
    run.

    """
    pass


@esat.command()
@click.argument("project_directory", type=click.Path())
def setup(project_directory):
    """
    Create the configuration file for a Batch source apportionment run in the provided directory.

    Parameters

    project_directory : The project directory where all output files are saved.

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

@esat.group()
def analysis_input():
    """
    Display tables and plots for input analysis.
    """
    pass


@analysis_input.command()
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


@analysis_input.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.argument("feature_idx", type=int)
def plot_input_uncertainty(project_directory, feature_idx):
    """
    Display the Concentration/Uncertainty scatter plot for the specified feature.

    Parameters

    project_directory : The project directory containing .toml configuration files.
    feature_idx : The index of the input/uncertainty data feature, column in the dataset.

    """
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    dh.plot_data_uncertainty(feature_idx=feature_idx)


@analysis_input.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.argument("feature_idx1", type=int)
@click.argument("feature_idx2", type=int)
def plot_feature_data(project_directory, feature_idx1, feature_idx2):
    """
    Display the feature vs feature scatter plot.

    Parameters

    project_directory : The project directory containing .toml configuration files.
    feature_idx1 : The index of the first (x-axis) input data feature, column in the dataset.
    feature_idx2 : The index of the second (y-axis) input data feature, column in the dataset.

    """
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    dh.plot_feature_data(x_idx=feature_idx1, y_idx=feature_idx2)


@analysis_input.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.argument("feature_idx", type=int, nargs=-1)
def plot_feature_timeseries(project_directory, feature_idx):
    """
    Display the feature(s) timeseries.

    Parameters

    project_directory : The project directory containing .toml configuration files.
    feature_idx : The indices of the input data feature, columns in the dataset.

    """
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    dh.plot_feature_timeseries(feature_selection=list(feature_idx))


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
def run(project_directory):
    """
    Run a batch source apportionment run using the provided configuration file.

    Parameters

    project_directory : The project directory containing .toml configuration files.

    """
    config = get_config(project_directory=project_directory)
    dh = DataHandler(**config["data"])
    V, U = dh.get_data()

    sa_models = BatchSA(V=V, U=U, **config["parameters"])
    _ = sa_models.train()
    output_path = os.path.join(config["project"]["directory"], "output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    sa_models.save(batch_name=config["project"]["name"], output_directory=output_path, pickle_batch=True)
    sa_models.save(batch_name=config["project"]["name"], output_directory=output_path, pickle_batch=False,
                   pickle_model=False, header=dh.features)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
@click.option("--feature_idx", default=0, type=int)
def solution_residual_histogram(project_directory, selected_model, feature_idx):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Residual Histogram for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_residual_histogram(feature_idx=feature_idx)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
def solution_statistics(project_directory, selected_model):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Statistics. Model selected: {selected_i}")
    ma.calculate_statistics()
    logger.info(ma.statistics)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
@click.option("--feature_idx", default=0, type=int)
def solution_estimated_observed(project_directory, selected_model, feature_idx):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Estimated/Observed for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_estimated_observed(feature_idx=feature_idx)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
@click.option("--feature_idx", default=0, type=int)
def solution_estimated_timeseries(project_directory, selected_model, feature_idx):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Estimated Timeseries for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_estimated_timeseries(feature_idx=feature_idx)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
@click.option("--factor_idx", default=0, type=int)
def solution_factor_profile(project_directory, selected_model, factor_idx):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Profile for model: {selected_i}, factor index: {factor_idx}")
    ma.plot_factor_profile(factor_idx=factor_idx)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
def solution_factor_fingerprints(project_directory, selected_model):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Fingerprints for model: {selected_i}")
    ma.plot_factor_fingerprints()


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
@click.option("--factor_idx1", default=0, type=int)
@click.option("--factor_idx2", default=1, type=int)
def solution_g_space(project_directory, selected_model, factor_idx1, factor_idx2):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution G-Space Plot for model: {selected_i}, factor 1: {factor_idx1}, factor 2: {factor_idx2}")
    ma.plot_g_space(factor_1=factor_idx1, factor_2=factor_idx2)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
@click.option("--feature_idx", default=0, type=int)
def solution_factor_contributions(project_directory, selected_model, feature_idx):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Contributions for model: {selected_i}, feature index: {feature_idx}")
    ma.plot_factor_contributions(feature_idx=feature_idx)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
def solution_factor_composition(project_directory, selected_model):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    logger.info(f"ESAT Solution Factor Composition for model: {selected_i}")
    ma.plot_factor_composition()


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--selected_model", default=-1, type=int)
@click.option("--factor_idx", default=0, type=int)
@click.option("--feature_idx", default=-1, type=int)
def solution_factor_surface(project_directory, selected_model, factor_idx, feature_idx):
    ma, selected_i = get_model_analysis(project_directory=project_directory, selected_model=selected_model)
    factor_idx = factor_idx if factor_idx != -1 else None
    feature_idx = feature_idx if feature_idx != -1 else None
    logger.info(f"ESAT Solution Factor Surface for model: {selected_i}, factor index: {factor_idx}, "
                f"feature index: {feature_idx}")
    ma.plot_factor_surface(feature_idx=feature_idx, factor_idx=factor_idx)


@esat.command()
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


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--disp", prompt=True, prompt_required=False, default=False,
              help="Run the displacement (DISP) error estimation method.")
@click.option("--bs", prompt=True, prompt_required=False, default=False,
              help="Run the bootstrap (BS) error estimation method.")
@click.option("--bsdisp", prompt=True, prompt_required=False, default=False,
              help="Run the bootstrap-displacement (BSDISP) error estimation method.")
def run_error(project_directory, disp, bs, bsdisp):
    """
    Run one or more of the error estimation methods using the specified error configuration file.

    Parameters

    project_directory : The project directory containing .toml configuration files.
    disp : Run the displacement (DISP) error estimation method.
    bs : Run the bootstrap (BS) error estimation method.
    bsdisp : Run the bootstrap-displacement (BSDISP) error estimation method.
    """
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
        features = e_config["disp"]["features"].strip('][').split(",")
        features = [int(f) for f in features] if len(features) > 1 else None
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
        features = e_config["bsdisp"]["features"].strip('][').split(",")
        features = [int(f) for f in features] if len(features) > 1 else None
        features_label = features if features is not None else "all"
        logger.info(f"Running BSDISP on model {selected_i}, features: {features_label}")
        _bs = None
        if e_config["bsdisp"]["bootstrap_output"] != "":
            bs_output_file = e_config["bsdisp"]["bootstrap_output"]
            if os.path.exists(bs_output_file):
                _bs = Bootstrap.load(bs_output_file)
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
        bsdisp.run()
        bsdisp.save(bsdisp_name=f"bsdisp-{config['project']['name']}", output_directory=error_dir, pickle_result=True)
        bsdisp.save(bsdisp_name=f"bsdisp-{config['project']['name']}", output_directory=error_dir, pickle_result=False)
        logger.info("BSDISP Completed")


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--disp", prompt=True, prompt_required=False, default=False,
              help="Summarize the displacement (DISP) error estimation results.")
@click.option("--bs", prompt=True, prompt_required=False, default=False,
              help="Summarize the bootstrap (BS) error estimation results.")
@click.option("--bsdisp", prompt=True, prompt_required=False, default=False,
              help="Summarize the bootstrap-displacement (BSDISP) error estimation results.")
def error_summary(project_directory, disp, bs, bsdisp):
    error_solution = get_error_model(
        project_directory=project_directory,
        disp=disp,
        bs=bs,
        bsdisp=bsdisp
    )
    error_solution.summary()


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--disp", prompt=True, prompt_required=False, default=False,
              help="Summarize the displacement (DISP) error estimation results.")
@click.option("--bs", prompt=True, prompt_required=False, default=False,
              help="Summarize the bootstrap (BS) error estimation results.")
@click.option("--bsdisp", prompt=True, prompt_required=False, default=False,
              help="Summarize the bootstrap-displacement (BSDISP) error estimation results.")
@click.option("--factor_idx", default=0, type=int)
def error_results(project_directory, disp, bs, bsdisp, factor_idx):
    error_solution = get_error_model(
        project_directory=project_directory,
        disp=disp,
        bs=bs,
        bsdisp=bsdisp
    )
    error_solution.plot_results(factor=factor_idx)


@esat.command()
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


@esat.command()
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


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_results(project_directory):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.display_results()


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--qtype", prompt=True, prompt_required=False, default='Aux',
              help="Plot the loss value for qtype: True, Robust, Aux")
def constrained_plot_q(project_directory, qtype):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_Q(Qtype=qtype)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_evaluate_constraints(project_directory):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.evaluate_constraints()


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_evaluate_expressions(project_directory):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.evaluate_expressions()


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--factor_idx", default=0, type=int)
def constrained_profile_contributions(project_directory, factor_idx):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_profile_contributions(factor_idx=factor_idx)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
def constrained_factor_fingerprints(project_directory):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_factor_fingerprints()


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--factor_idx1", default=0, type=int)
@click.option("--factor_idx2", default=0, type=int)
@click.option("--show_base", default=True, type=bool)
@click.option("--show_delta", default=True, type=bool)
def constrained_g_space(project_directory, factor_idx1, factor_idx2, show_base, show_delta):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_g_space(factor_idx1=factor_idx1, factor_idx2=factor_idx2, show_base=show_base, show_delta=show_delta)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True))
@click.option("--feature_idx", default=0, type=int)
def constrained_profile_contributions(project_directory, feature_idx):
    c_model = get_constrained_model(project_directory=project_directory)
    c_model.plot_factor_contributions(feature_idx=feature_idx)


if __name__ == "__main__":
    esat()
