import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import logging
import pandas as pd
from esat.data.datahandler import DataHandler

logger = logging.getLogger(__name__)


class TestDataHandler:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    batch_name = "bs_test00"

    @classmethod
    def setup_class(self):
        logger.info("Running SA Test Setup")
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
        self.input_file = os.path.join(self.data_path, "Dataset-BatonRouge-con.csv")
        self.uncertainty_file = os.path.join(self.data_path, "Dataset-BatonRouge-unc.csv")

    def test_load(self):
        datahandler = DataHandler(
            input_path=self.input_file,
            uncertainty_path=self.uncertainty_file,
            index_col='Date'
        )
        V, U = datahandler.get_data()
        assert V.shape == (307, 41)

    def test_load_dataframe(self):
        input_df = pd.read_csv(self.input_file, index_col="Date")
        uncertainty_df = pd.read_csv(self.uncertainty_file, index_col="Date")
        datahandler = DataHandler.load_dataframe(input_df=input_df, uncertainty_df=uncertainty_df)
        V, U = datahandler.get_data()
        assert V.shape == (307, 41)
