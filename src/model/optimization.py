from src.model.model import NMFModel
from src.model.base_nmf import BaseSearch
from src.data.datahandler import DataHandler


class ComponentSearch:

    def __init__(self, seed, datahandler: DataHandler, ):
        self.seed = seed
        self.datahandler = datahandler
        self.results = None
        self.Q = []
        self.min_n = 0
        self.max_n = 0

    def search(self, min_component: int = 2, max_component: int = 10,
               max_iterations: int = 1000, epochs: int = 1,
               converge_diff: int = 10, converge_iter=100):
        self.results = {}
        self.min_n = min_component
        self.max_n = max_component
        for n in range(min_component, max_component):
            model = BaseSearch(n_components=n,
                               V=self.datahandler.input_data_processed,
                               U=self.datahandler.uncertainty_data_processed,
                               method="kl",
                               seed=self.seed,
                               epochs=epochs,
                               max_iterations=max_iterations,
                               converge_delta=converge_diff,
                               converge_n=converge_iter
                               )
            model.parallel_train()
            self.results[n] = model.results[0]
            self.Q.append(model.results[0]["Q"])
