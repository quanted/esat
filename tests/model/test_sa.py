import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import numpy as np
import logging
from esat.model.sa import SA
from esat.data.datahandler import DataHandler

logger = logging.getLogger(__name__)


class TestSA:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    model_name = "sa_test00"

    @classmethod
    def setup_class(self):
        logger.info("Running SA Test Setup")
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
        self.input_file = os.path.join(self.data_path, "Dataset-BatonRouge-con.csv")
        self.uncertainty_file = os.path.join(self.data_path, "Dataset-BatonRouge-unc.csv")
        self.datahandler = DataHandler(
            input_path=self.input_file,
            uncertainty_path=self.uncertainty_file,
            index_col='Date'
        )
        self.V, self.U = self.datahandler.get_data()

    @classmethod
    def teardown_class(self):
        save_path = os.path.join(self.data_path, "output")
        for _file in os.listdir(save_path):
            if self.model_name in str(_file):
                if os.path.exists(_file):
                    os.remove(_file)

    def test_initialization(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n)
        sa.initialize()
        assert sa.H is not None
        assert sa.W is not None
        assert sa.W.shape == (self.V.shape[0], factor_n)
        assert sa.H.shape == (factor_n, self.V.shape[1])

    def test_initialization_kmeans(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n)
        sa.initialize(init_method='kmeans')
        assert sa.H is not None
        assert sa.W is not None
        assert sa.W.shape == (self.V.shape[0], factor_n)
        assert sa.H.shape == (factor_n, self.V.shape[1])

    def test_initialization_cmeans(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n)
        sa.initialize(init_method='cmeans', fuzziness=5.0)
        assert sa.H is not None
        assert sa.W is not None
        assert sa.W.shape == (self.V.shape[0], factor_n)
        assert sa.H.shape == (factor_n, self.V.shape[1])

    def test_initialization_H(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n)
        _H = np.ones(shape=(1, self.V.shape[1]))
        sa.initialize(H=_H)
        assert sa.H is not None
        assert sa.W is not None
        assert sa.W.shape == (self.V.shape[0], factor_n)
        assert sa.H.shape == (factor_n, self.V.shape[1])
        assert np.sum(sa.H[0] - _H) == 0.0

    def test_ls_nmf(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n, method="ls-nmf")
        sa.initialize()
        sa.train(max_iter=500, converge_delta=1.0, converge_n=10)
        assert sa.WH is not None
        assert sa.Qtrue is not None

        sa2 = SA(V=self.V, U=self.U, factors=factor_n, method="ls-nmf", optimized=False)
        sa2.initialize()
        sa2.train(max_iter=500, converge_delta=1.0, converge_n=10)
        assert sa2.WH is not None
        assert sa2.Qtrue is not None

    def test_ws_nmf(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n, method="ws-nmf")
        sa.initialize()
        sa.train(max_iter=500, converge_delta=1.0, converge_n=10)
        assert sa.WH is not None
        assert sa.Qtrue is not None

    def test_save(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n)
        sa.initialize()
        sa.train(max_iter=500, converge_delta=1.0, converge_n=10)
        save_path = os.path.join(self.data_path, "output")
        saved_file = sa.save(
            model_name=self.model_name,
            output_directory=save_path,
            header=self.datahandler.features
        )
        assert str(saved_file) == str(save_path)
        saved_file_pkl = sa.save(
            model_name=self.model_name,
            output_directory=save_path,
            pickle_model=True,
            header=self.datahandler.features
        )
        assert str(saved_file_pkl) == str(os.path.join(save_path, f"{self.model_name}.pkl"))

    def test_load(self):
        save_path = os.path.join(self.data_path, "output")
        save_file = os.path.join(save_path, f"{self.model_name}.pkl")
        sa = SA.load(file_path=save_file)
        assert sa is not None



