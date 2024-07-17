import sys
import os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import numpy as np
import logging
from esat.model.ls_nmf import LSNMF
from esat.model.sa import SA
from esat.metrics import q_loss
from esat.data.datahandler import DataHandler

logger = logging.getLogger(__name__)


class TestLSNMF:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    model_name = "sa_test00"

    @classmethod
    def setup_class(self):
        logger.info("Running LS-NMF Test Setup")
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
        self.input_file = os.path.join(self.data_path, "Dataset-BatonRouge-con.csv")
        self.uncertainty_file = os.path.join(self.data_path, "Dataset-BatonRouge-unc.csv")
        self.datahandler = DataHandler(
            input_path=self.input_file,
            uncertainty_path=self.uncertainty_file,
            index_col='Date'
        )
        self.V, self.U = self.datahandler.get_data()

    def test_initialization(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n, seed=42)
        sa.initialize()
        _W, _H = LSNMF.update(V=sa.V, We=sa.We, W=sa.W, H=sa.H)
        assert _W.shape == (self.V.shape[0], factor_n)
        assert _H.shape == (factor_n, self.V.shape[1])

    def test_decreasing_loss(self):
        factor_n = 6
        sa = SA(V=self.V, U=self.U, factors=factor_n)
        sa.initialize(init_method='kmeans')
        decreasing = True
        q_h = float("inf")
        _W = sa.W
        _H = sa.H
        for i in range(100):
            _W, _H = LSNMF.update(V=sa.V, We=sa.We, W=_W, H=_H)
            q_i = q_loss(V=sa.V, U=sa.U, W=_W, H=_H)
            if q_i > q_h:
                decreasing = False
            q_h = q_i
        assert decreasing