import logging
import numpy as np
from esat.model.batch_sa import BatchSA
from esat.metrics import cal_cophenetic, cal_dispersion, cal_connectivity
from numpy import linalg as LA

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


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
                 converge_n=10,
                 parallel: bool = False,
                 optimized: bool = False,
                 verbose: bool = True
                 ):
        self.seed = seed
        self.data = data
        self.uncertainty = uncertainty
        self.results = None
        self.Qtrue = []
        self.Qrobust = []
        self.Cophen = []
        self.Disp = []
                     
        # Bayes Information metrics from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9181460/
        self.BIC1 = []
        self.BIC2 = []
        self.BIC3 = []
                     
        self.min_n = min_factor
        self.max_n = max_factor
        self.method = method
        self.max_iter = max_iterations
        self.models = models
        self.converge_delta = converge_delta
        self.converge_n = converge_n
        self.optimized = optimized
        self.parallel = parallel
        self.verbose = verbose

    def search(self):
        self.results = {}
        for factors in range(self.min_n, self.max_n+1):
            logger.info(f"Factor search - factors: {factors}, models: {self.models}, method: {self.method}")
            model = BatchSA(
                V=self.data,
                U=self.uncertainty,
                factors=factors,
                method=self.method,
                seed=self.seed,
                max_iter=self.max_iter,
                models=self.models,
                converge_n=self.converge_n,
                converge_delta=self.converge_delta,
                parallel=self.parallel,
                optimized=self.optimized,
                verbose=self.verbose
            )

            s = float(np.shape(self.data)[0])
            f = float(np.shape(self.data)[1])
            
            model.train()
            self.results[factors] = model
            self.Qtrue.append(model.results[model.best_model].Qtrue)
            self.Qrobust.append(model.results[model.best_model].Qrobust)
            self.Cophen.append(cal_cophenetic(model.results[model.best_model].WH))
            self.Disp.append(cal_dispersion(model.results[model.best_model].WH))

            Vp = model.results[model.best_model].WH
            C1 = (s+f)/(s*f)
            C2 = factors*C1
            C3 = min(s**0.5,f**0.5)**2
            norm = (LA.norm(Vp-self.data))**2
            
            self.BIC1.append(np.log10(norm)+C2*np.log10(1/C1))
            self.BIC2.append(np.log10(norm)+C2*np.log10(C3))
            self.BIC3.append(np.log10(norm)+C2*np.log10(C3)/C3)
