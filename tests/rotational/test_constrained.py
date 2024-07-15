import sys, os
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(src_path)
import logging
from esat.model.batch_sa import BatchSA
from esat.data.datahandler import DataHandler
from esat.rotational.constrained import ConstrainedModel

logger = logging.getLogger(__name__)


class TestBatchSA:

    data_path = None
    input_file = None
    uncertainty_file = None
    datahandler = None
    V = None
    U = None
    cm_name = "cm_test00"
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

    def test_add_constraint(self):
        cm = ConstrainedModel(base_model=self.batch.results[1], data_handler=self.datahandler, softness=1.0)
        added = cm.add_constraint(constraint_type="set to zero", index=(0, 3), target="feature")
        assert added
        assert len(cm.constraints) == 1
        added2 = cm.add_constraint(constraint_type="define limits", index=(2,10), target="feature", min_value=0.1,
                                   max_value=0.9)
        assert added2
        assert len(cm.constraints) == 2

    def test_remove_constraint(self):
        cm = ConstrainedModel(base_model=self.batch.results[1], data_handler=self.datahandler, softness=1.0)
        added = cm.add_constraint(constraint_type="set to zero", index=(0, 3), target="feature")
        assert added
        cm.remove_constraint(constraint_label='factor:0|feature:3')
        assert len(cm.constraints) == 0

    def test_add_expression(self):
        cm = ConstrainedModel(base_model=self.batch.results[1], data_handler=self.datahandler, softness=1.0)
        expression1 = "(0.66*[factor:1|feature:2])-(4.2*[factor:2|feature:4])=0,250"
        added = cm.add_expression(expression1)
        assert added
        assert len(cm.expressions) == 1
        expression2 = "(0.35*[factor:0|feature:3])-(2.0*[factor:1|feature:3])-(3.7*[factor:3|feature:4])=0,250"
        added2 = cm.add_expression(expression2)
        assert added2
        assert len(cm.expressions) == 2

    def test_remove_expression(self):
        cm = ConstrainedModel(base_model=self.batch.results[1], data_handler=self.datahandler, softness=1.0)
        expression1 = "(0.66*[factor:1|feature:2])-(4.2*[factor:2|feature:4])=0,250"
        added = cm.add_expression(expression1)
        assert added
        cm.remove_expression(0)
        assert len(cm.expressions) == 0

    def test_run(self):
        cm = ConstrainedModel(base_model=self.batch.results[1], data_handler=self.datahandler, softness=1.0)
        added = cm.add_constraint(constraint_type="set to zero", index=(0, 3), target="feature")
        expression1 = "(0.66*[factor:1|feature:2])-(4.2*[factor:2|feature:4])=0,250"
        added = cm.add_expression(expression1)
        cm.train(max_iterations=500)

    def test_save(self):
        cm = ConstrainedModel(base_model=self.batch.results[1], data_handler=self.datahandler, softness=1.0)
        added = cm.add_constraint(constraint_type="set to zero", index=(0, 3), target="feature")
        expression1 = "(0.66*[factor:1|feature:2])-(4.2*[factor:2|feature:4])=0,250"
        added = cm.add_expression(expression1)
        cm.train(max_iterations=500)

        save_path = os.path.join(self.data_path, "test_output")
        saved_file = cm.save(
            model_name=self.cm_name,
            output_directory=save_path,
            pickle_model=True
        )
        output_name = f"constrained_model-{self.cm_name}.pkl"
        assert os.path.exists(str(os.path.join(save_path, f"{output_name}")))

    def test_load(self):
        output_name = f"constrained_model-{self.cm_name}.pkl"
        save_path = os.path.join(self.data_path, "test_output")
        save_file = os.path.join(save_path, output_name)
        cm = ConstrainedModel.load(file_path=save_file)
        assert cm is not None




