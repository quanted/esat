import os
import configparser
import esat.cli.esat_cli as cli
from click.testing import CliRunner

project_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "test_output",
                                 "cli_test")
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def test_setup():
    runner = CliRunner()
    result = runner.invoke(cli.setup, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "run_config.toml"))


def test_run():
    run_config_file = os.path.join(project_directory, "run_config.toml")
    input_file = os.path.join(data_path, "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join(data_path, "Dataset-BatonRouge-unc.csv")

    run_config = configparser.ConfigParser()
    run_config.read(run_config_file)
    run_config["project"]["name"] = "cli_test"
    run_config["data"]["input_path"] = input_file
    run_config["data"]["uncertainty_path"] = uncertainty_file
    run_config["data"]["index_col"] = "Date"
    run_config["parameters"]["factors"] = "6"
    run_config["parameters"]["models"] = "2"
    run_config["parameters"]["max_iter"] = "500"
    run_config["parameters"]["converge_delta"] = "0.1"
    run_config["parameters"]["optimized"] = "False"
    run_config["parameters"]["parallel"] = "False"
    with open(run_config_file, 'w') as cfile:
        run_config.write(cfile)
    runner = CliRunner()
    result = runner.invoke(cli.run, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "output", "cli_test.pkl"))


def test_setup_sim():
    runner = CliRunner()
    result = runner.invoke(cli.setup_sim, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "sim_config.toml"))


def test_gen_sim():
    runner = CliRunner()
    result = runner.invoke(cli.generate_sim, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "esat_simulator.pkl"))


def test_run_sim():
    run_config_file = os.path.join(project_directory, "run_config.toml")

    run_config = configparser.ConfigParser()
    run_config.read(run_config_file)
    run_config["project"]["name"] = "cli_sim_test"
    run_config["data"]["input_path"] = ""
    run_config["data"]["uncertainty_path"] = ""
    run_config["data"]["index_col"] = "Date"
    run_config["parameters"]["factors"] = "6"
    run_config["parameters"]["models"] = "2"
    run_config["parameters"]["max_iter"] = "500"
    run_config["parameters"]["converge_delta"] = "0.1"
    run_config["parameters"]["optimized"] = "False"
    run_config["parameters"]["parallel"] = "False"
    with open(run_config_file, 'w') as cfile:
        run_config.write(cfile)

    runner = CliRunner()
    result = runner.invoke(cli.run, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "output", "cli_sim_test.pkl"))


def test_setup_error():
    runner = CliRunner()
    result = runner.invoke(cli.setup_error, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "error_config.toml"))


def test_run_error_fail():
    runner = CliRunner()
    result = runner.invoke(cli.run_error, [project_directory])
    assert result.exit_code == 0


def test_run_error_disp():
    run_config_file = os.path.join(project_directory, "run_config.toml")
    error_config_file = os.path.join(project_directory, "error_config.toml")
    input_file = os.path.join(data_path, "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join(data_path, "Dataset-BatonRouge-unc.csv")

    run_config = configparser.ConfigParser()
    run_config.read(run_config_file)
    run_config["project"]["name"] = "cli_test"
    run_config["data"]["input_path"] = input_file
    run_config["data"]["uncertainty_path"] = uncertainty_file
    run_config["data"]["index_col"] = "Date"
    run_config["parameters"]["factors"] = "6"
    run_config["parameters"]["models"] = "2"
    run_config["parameters"]["max_iter"] = "500"
    run_config["parameters"]["converge_delta"] = "0.1"
    run_config["parameters"]["optimized"] = "False"
    run_config["parameters"]["parallel"] = "False"
    run_config["parameters"]["verbose"] = "True"
    with open(run_config_file, 'w') as cfile:
        run_config.write(cfile)

    error_config = configparser.ConfigParser()
    error_config.read(error_config_file)
    error_config["project"]["selected_model"] = "1"
    error_config["disp"]["features"] = "[0]"
    with open(error_config_file, 'w') as cfile:
        error_config.write(cfile)

    runner = CliRunner()
    result = runner.invoke(cli.run_error, ["--disp", project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "error", f"disp-{run_config['project']['name']}.pkl"))


def test_run_error_bs():
    run_config_file = os.path.join(project_directory, "run_config.toml")
    error_config_file = os.path.join(project_directory, "error_config.toml")
    input_file = os.path.join(data_path, "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join(data_path, "Dataset-BatonRouge-unc.csv")

    run_config = configparser.ConfigParser()
    run_config.read(run_config_file)
    run_config["project"]["name"] = "cli_test"
    run_config["data"]["input_path"] = input_file
    run_config["data"]["uncertainty_path"] = uncertainty_file
    run_config["data"]["index_col"] = "Date"
    run_config["parameters"]["factors"] = "6"
    run_config["parameters"]["models"] = "2"
    run_config["parameters"]["max_iter"] = "500"
    run_config["parameters"]["converge_delta"] = "0.1"
    run_config["parameters"]["optimized"] = "False"
    run_config["parameters"]["parallel"] = "False"
    run_config["parameters"]["verbose"] = "True"
    with open(run_config_file, 'w') as cfile:
        run_config.write(cfile)

    error_config = configparser.ConfigParser()
    error_config.read(error_config_file)
    error_config["project"]["selected_model"] = "1"
    error_config["bs"]["bootstrap_n"] = "2"
    with open(error_config_file, 'w') as cfile:
        error_config.write(cfile)

    runner = CliRunner()
    result = runner.invoke(cli.run_error, ["--bs", project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "error", f"bs-{run_config['project']['name']}.pkl"))


def test_run_error_bsdisp():
    run_config_file = os.path.join(project_directory, "run_config.toml")
    error_config_file = os.path.join(project_directory, "error_config.toml")
    input_file = os.path.join(data_path, "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join(data_path, "Dataset-BatonRouge-unc.csv")

    run_config = configparser.ConfigParser()
    run_config.read(run_config_file)
    run_config["project"]["name"] = "cli_test"
    run_config["data"]["input_path"] = input_file
    run_config["data"]["uncertainty_path"] = uncertainty_file
    run_config["data"]["index_col"] = "Date"
    run_config["parameters"]["factors"] = "6"
    run_config["parameters"]["models"] = "2"
    run_config["parameters"]["max_iter"] = "500"
    run_config["parameters"]["converge_delta"] = "0.1"
    run_config["parameters"]["optimized"] = "False"
    run_config["parameters"]["parallel"] = "False"
    run_config["parameters"]["verbose"] = "True"
    with open(run_config_file, 'w') as cfile:
        run_config.write(cfile)

    error_config = configparser.ConfigParser()
    error_config.read(error_config_file)
    error_config["project"]["selected_model"] = "1"
    error_config["bsdisp"]["bootstrap_output"] = os.path.join(project_directory, "error",
                                                              f"bs-{run_config['project']['name']}.pkl")
    error_config["bsdisp"]["features"] = "[0]"
    with open(error_config_file, 'w') as cfile:
        error_config.write(cfile)

    runner = CliRunner()
    result = runner.invoke(cli.run_error, ["--bsdisp", project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "error", f"bsdisp-{run_config['project']['name']}.pkl"))


def test_setup_constrained():
    runner = CliRunner()
    result = runner.invoke(cli.setup_constrained, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(project_directory, "constrained_config.toml"))


def test_run_constrained():
    run_config_file = os.path.join(project_directory, "run_config.toml")
    run_config = configparser.ConfigParser()
    run_config.read(run_config_file)

    constrained_config_file = os.path.join(project_directory, "constrained_config.toml")
    constrained_config = configparser.ConfigParser()
    constrained_config.read(constrained_config_file)

    constrained_config["constraints"] = {
        "constraint1": '{"constraint_type":"set to zero","index":[0,1],"target":"feature"}',
        "constraint2": '{"constraint_type":"define limits","index":[2,4],"target":"feature","min_value":0,"max_value":1}',
        "constraint3": '{"constraint_type":"pull up","index":[5,1],"target":"feature","dQ":50}',
        "constraint4": '{"constraint_type":"pull down","index":[2,0],"target":"feature","dQ":50}',
        "constraint5": '{"constraint_type":"set to base value","index":[10,2],"target":"feature"}',
        "constraint6": '{"constraint_type":"pull to value","index":[21,5],"target":"feature","target_value":0,"dQ":50}'
    }
    constrained_config["expressions"] = {
        "expression1": "(0.66*[factor:2|feature:1])-(4.2*[factor:2|feature:9])=0,250",
        "expression2": "(0.35*[factor:3|feature:3])-(2.0*[factor:3|feature:11])-(3.7*[factor:5|feature:20])=0,250",
    }
    with open(constrained_config_file, 'w') as cfile:
        constrained_config.write(cfile)

    constrained_dir = os.path.join(project_directory, "constrained")
    constrained_file = os.path.join(constrained_dir, f"constrained_model-{run_config['project']['name']}.pkl")

    runner = CliRunner()
    result = runner.invoke(cli.run_constrained, [project_directory])
    assert result.exit_code == 0
    assert os.path.exists(constrained_file)


