import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import logging
from esat.model.batch_sa import BatchSA
from esat.data.datahandler import DataHandler

logger = logging.getLogger(__name__)


class TestBatchSA:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    batch_name = "batch_test00"

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

    def test_ls_nmf(self):
        factor_n = 6
        models = 2
        bs = BatchSA(V=self.V, U=self.U, models=models, factors=factor_n, method="ls-nmf",
                     max_iter=500, converge_delta=1.0, converge_n=10, parallel=False)
        bs.train()
        assert len(bs.results) == 2
        assert bs.best_model is not None

    def test_py(self):
        factor_n = 6
        models = 2
        bs = BatchSA(V=self.V, U=self.U, models=models, factors=factor_n, method="ls-nmf",
                     max_iter=500, converge_delta=1.0, converge_n=10, parallel=False)
        bs.train()
        assert len(bs.results) == 2
        assert bs.best_model is not None

    def test_ws_nmf(self):
        factor_n = 6
        models = 2
        bs = BatchSA(V=self.V, U=self.U, models=models, factors=factor_n, method="ws-nmf",
                     max_iter=500, converge_delta=1.0, converge_n=10, parallel=False)
        bs.train()
        assert len(bs.results) == 2
        assert bs.best_model is not None

    def test_save(self):
        factor_n = 6
        models = 2
        bs = BatchSA(V=self.V, U=self.U, models=models, factors=factor_n, method="ls-nmf",
                     max_iter=500, converge_delta=1.0, converge_n=10, parallel=False)
        bs.train()
        save_path = os.path.join(self.data_path, "test_output")
        saved_file = bs.save(
            batch_name=self.batch_name,
            output_directory=save_path,
            pickle_batch=False,
            header=self.datahandler.features
        )
        assert str(saved_file) == str(save_path)
        saved_file_pkl = bs.save(
            batch_name=self.batch_name,
            output_directory=save_path,
            pickle_batch=True,
            header=self.datahandler.features
        )
        assert os.path.exists(str(os.path.join(save_path, f"{self.batch_name}.pkl")))

    def test_load(self):
        save_path = os.path.join(self.data_path, "test_output")
        save_file = os.path.join(save_path, f"{self.batch_name}.pkl")
        bs = BatchSA.load(file_path=save_file)
        assert bs is not None




