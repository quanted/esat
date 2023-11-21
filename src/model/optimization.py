import logging
from src.model.batch_nmf import BatchNMF

logger = logging.getLogger("NMF")
logger.setLevel(logging.DEBUG)


class FactorSearch:

    def __init__(self, seed, data, uncertainty, ):
        self.seed = seed
        self.data = data
        self.uncertainty = uncertainty
        self.results = None
        self.Q = []
        self.min_n = 0
        self.max_n = 0

    def search(self, min_factor: int = 2, max_factor: int = 10,
               method: str = 'ls-nmf',
               max_iterations: int = 1000, models: int = 10,
               converge_delta: int = 0.1, converge_n=10):
        self.results = {}
        self.min_n = min_factor
        self.max_n = max_factor
        for n in range(min_factor, max_factor+1):
            logger.info(f"Factor search - factors: {n}, models: {models}, method: {method}")
            model = BatchNMF(
                V=self.data,
                U=self.uncertainty,
                factors=n,
                method=method,
                seed=self.seed,
                max_iter=max_iterations,
                models=models,
                converge_n=converge_n,
                converge_delta=converge_delta,
                parallel=True,
                optimized=True,
                verbose=True
            )
            model.train()
            self.results[n] = model.results[0]
            self.Q.append(model.results[0]["Q"])


