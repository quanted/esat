import os
import shutil
import pytest

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    def remove_files():
        save_path = os.path.join(data_path, "test_output")
        shutil.rmtree(save_path)
    request.addfinalizer(remove_files)


def pytest_configure(config):
    save_path = os.path.join(data_path, "test_output")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
