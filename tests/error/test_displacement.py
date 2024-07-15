import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import logging
from esat.error.displacement import Displacement
from esat.model.batch_sa import BatchSA
from esat.data.datahandler import DataHandler

logger = logging.getLogger(__name__)


class TestDisplacement:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    disp_name = "disp_test00"
    batch = None

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
        self.batch = BatchSA(V=self.V, U=self.U, models=2, factors=6, method="ls-nmf",
                             max_iter=500, converge_delta=1.0, converge_n=10, parallel=False)
        self.batch.train()
        self.disp = None

    def test_run(self):
        selected_model = 1
        disp = Displacement(sa=self.batch.results[selected_model], feature_labels=self.datahandler.features,
                            features=[0])
        disp.run()
        assert disp.compiled_results is not None
        assert len(disp.increase_results) == 6

    def test_save(self):
        selected_model = 1
        disp = Displacement(sa=self.batch.results[selected_model], feature_labels=self.datahandler.features,
                            features=[0])
        disp.run()
        save_path = os.path.join(self.data_path, "test_output")
        saved_file = disp.save(
            disp_name=self.disp_name,
            output_directory=save_path,
        )
        assert os.path.exists(str(os.path.join(save_path, f"{self.disp_name}.pkl")))

    def test_load(self):
        save_path = os.path.join(self.data_path, "test_output")
        save_file = os.path.join(save_path, f"{self.disp_name}.pkl")
        disp = Displacement.load(file_path=save_file)
        assert disp is not None
