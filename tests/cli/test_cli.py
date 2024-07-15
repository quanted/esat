import os
import esat.cli.esat_cli as cli
from click.testing import CliRunner

project_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "test_output",
                                 "cli_test")


def test_setup():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.setup, [project_directory])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(project_directory, "run_config.toml"))


def test_setup_sim():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.setup_sim, [project_directory])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(project_directory, "sim_config.toml"))


def test_setup_error():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.setup_error, [project_directory])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(project_directory, "error_config.toml"))


def test_setup_constrained():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.setup_constrained, [project_directory])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(project_directory, "constrained_config.toml"))
