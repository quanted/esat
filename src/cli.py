import os
import click
import configparser
import logging
from src.data.datahandler import DataHandler
from src.model.batch_sa import BatchSA
from src.error.displacement import Displacement
from src.error.bootstrap import Bootstrap
from src.error.bs_disp import BSDISP
from src.configs import run_config, error_config

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def esat():
    pass


@esat.command()
@click.argument("project_directory", type=click.Path(), help="The project directory where all output files are saved.")
def setup(project_directory):
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)
    new_config = run_config
    new_config['project']['directory'] = project_directory
    new_config_file = os.path.join(project_directory, "run_config.toml")
    with open(new_config_file, 'w') as configfile:
        new_config.write(configfile)
    logger.info(f"New run configuration file created. File path: {new_config_file}")


@esat.command()
@click.argument("config_file", type=click.Path(exists=True),
                help="The source apportionment configuration file used for generating SA solutions.")
def run(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

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
@click.argument("project_directory", type=click.Path(exists=True),
                help="The project directory where all output files are saved.")
@click.argument("method", help="The error estimation method to create a new config file for.")
def error_setup(project_directory):
    new_config = error_config
    new_config_file = os.path.join(project_directory, "error_config.toml")
    with open(new_config_file, 'w') as configfile:
        new_config.write(configfile)
    logger.info(f"New error configuration file created. File path: {new_config_file}")


@esat.command()
@click.argument("run_config_file", type=click.Path(exists=True),
                help="The source apportionment configuration file used for generating SA solutions.")
@click.argument("config_file", type=click.Path(exists=True), help="The error estimation configuration file.")
@click.option("disp", prompt=True, prompt_required=False, default=False,
              help="Run the displacement (DISP) error estimation method.")
@click.option("bs", prompt=True, prompt_required=False, default=False,
              help="Run the bootstrap (BS) error estimation method.")
@click.option("bsdisp", prompt=True, prompt_required=False, default=False,
              help="Run the bootstrap-displacement (BSDISP) error estimation method.")
def run_error(run_config_file, config_file, disp, bs, bsdisp):
    config = configparser.ConfigParser()
    config.read(run_config_file)

    error_config = configparser.ConfigParser()
    error_config.read(config_file)

    dh = DataHandler(**config["data"])

    batch_pkl = os.path.join(config["project"]["directory"], "output", f"{config['project']['name']}.pkl")
    sa_models = BatchSA.load(file_path=batch_pkl)
    error_dir = os.path.join(config["project"]["directory"], "error")
    bs = None
    if disp:
        selected_i = int(error_config["disp"]["selected_model"])
        selected_i = sa_models.best_model if selected_i == -1 else selected_i
        selected_model = sa_models.results[selected_i]
        features = error_config["disp"]["features"].strip('][').split(",")
        disp = Displacement(
            sa=selected_model,
            feature_labels=dh.features,
            model_selected=selected_i,
            features=features
        )
        disp.run()
        disp.save(disp_name=config["project"]["name"], output_directory=error_dir, pickle_result=True)
        disp.save(disp_name=config["project"]["name"], output_directory=error_dir, pickle_result=False)
    if bs:
        selected_i = int(error_config["bs"]["selected_model"])
        selected_i = sa_models.best_model if selected_i == -1 else selected_i
        selected_model = sa_models.results[selected_i]
        block_size = int(error_config["bs"]["block_size"])
        block_size = dh.optimal_block if block_size == -1 else block_size
        seed = int(error_config["bs"]["seed"])
        seed = selected_model.seed if seed == -1 else seed
        bs = Bootstrap(
            sa=selected_model,
            feature_labels=dh.features,
            model_selected=selected_i,
            bootstrap_n=int(error_config["bs"]["bootstrap_n"]),
            block_size=block_size,
            threshold=float(error_config["bs"]["threshold"]),
            seed=seed
        )
        bs.run()
        bs.save(bs_name=config["project"]["name"], output_directory=error_dir, pickle_result=True)
        bs.save(bs_name=config["project"]["name"], output_directory=error_dir, pickle_result=False)
    if bsdisp:
        selected_i = int(error_config["bsdisp"]["selected_model"])
        selected_i = sa_models.best_model if selected_i == -1 else selected_i
        selected_model = sa_models.results[selected_i]
        block_size = int(error_config["bsdisp"]["block_size"])
        block_size = dh.optimal_block if block_size == -1 else block_size
        seed = int(error_config["bsdisp"]["seed"])
        seed = selected_model.seed if seed == -1 else seed

        if error_config["bsdisp"]["bootstrap_output"] != "":
            bs_output_file = error_config["bsdisp"]["bootstrap_output"]
            if os.path.exists(bs_output_file):
                bs = Bootstrap.load(bs_output_file)

        bsdisp = BSDISP(
            sa=selected_model,
            bootstrap=bs,
            feature_labels=dh.features,
            model_selected=selected_i,
            bootstrap_n=int(error_config["bsdisp"]["bootstrap_n"]),
            block_size=block_size,
            threshold=float(error_config["bsdisp"]["threshold"]),
            seed=seed,
            max_search=int(error_config["bsdisp"]["max_search"]),
            threshold_dQ=float(error_config["bsdisp"]["threshold_dQ"])
        )
        bsdisp.run()
        bsdisp.save(bsdisp_name=config["project"]["name"], output_directory=error_dir, pickle_result=True)
        bsdisp.save(bsdisp_name=config["project"]["name"], output_directory=error_dir, pickle_result=False)


@esat.command()
@click.argument("project_directory", type=click.Path(exists=True),
                help="The project directory where all output files are saved.")
def constrained_setup(project_directory):
    pass


@esat.command()
@click.argument("run_config_file", type=click.Path(exists=True),
                help="The source apportionment configuration file used for generating SA solutions.")
@click.argument("config_file", type=click.Path(exists=True), help="The constrained model run configuration file.")
def run_constrained(config_file):
    pass
