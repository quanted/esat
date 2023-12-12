import logging
from src.model.batch_nmf import BatchNMF

logger = logging.getLogger("NMF")
logger.setLevel(logging.DEBUG)


class FactorSearch:

    def __init__(self,
                 seed,
                 data,
                 uncertainty,
                 min_factor: int = 2,
                 max_factor: int = 10,
                 method: str = 'ls-nmf',
                 max_iterations: int = 1000,
                 models: int = 10,
                 converge_delta: int = 0.1,
                 converge_n=10
                 ):
        self.seed = seed
        self.data = data
        self.uncertainty = uncertainty
        self.results = None
        self.Q = []

        self.min_n = min_factor
        self.max_n = max_factor
        self.method = method
        self.max_iter = max_iterations
        self.models = models
        self.converge_delta = converge_delta
        self.converge_n = converge_n

    def search(self):
        self.results = {}
        for n in range(self.min_n, self.max_n+1):
            logger.info(f"Factor search - factors: {n}, models: {self.models}, method: {self.method}")
            model = BatchNMF(
                V=self.data,
                U=self.uncertainty,
                factors=n,
                method=self.method,
                seed=self.seed,
                max_iter=self.max_iter,
                models=self.models,
                converge_n=self.converge_n,
                converge_delta=self.converge_delta,
                parallel=True,
                optimized=True,
                verbose=True
            )
            model.train()
            self.results[n] = model
            self.Q.append(model.results[model.best_model].Qtrue)
