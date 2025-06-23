import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import logging
from esat.error.bootstrap import Bootstrap
from esat.model.batch_sa import BatchSA
from esat.data.datahandler import DataHandler

logger = logging.getLogger(__name__)


class TestBootstrap:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    bs_name = "bs_test00"
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

    def test_run(self):
        selected_model = 1
        bs = Bootstrap(sa=self.batch.results[selected_model], feature_labels=self.datahandler.features,
                       model_selected=selected_model, bootstrap_n=2, block_size=self.datahandler.optimal_block,
                       parallel=False, threshold=0.6, seed=42)
        bs.run()
        assert bs.mapping_df is not None
        assert len(bs.bs_results) == 2

    def test_save(self):
        selected_model = 1
        bs = Bootstrap(sa=self.batch.results[selected_model], feature_labels=self.datahandler.features,
                       model_selected=selected_model, bootstrap_n=2, block_size=self.datahandler.optimal_block,
                       parallel=False, threshold=0.6, seed=42)
        bs.run()
        save_path = os.path.join(self.data_path, "test_output")
        saved_file = bs.save(
            bs_name=self.bs_name,
            output_directory=save_path
        )
        assert os.path.exists(str(os.path.join(save_path, f"{self.bs_name}.pkl")))

    def test_load(self):
        save_path = os.path.join(self.data_path, "test_output")
        save_file = os.path.join(save_path, f"{self.bs_name}.pkl")
        bs = Bootstrap.load(file_path=save_file)
        assert bs is not None
