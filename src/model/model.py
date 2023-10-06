import sys
import os
module_path = os.path.abspath(os.path.join('..', "nmf_py"))
sys.path.append(module_path)

import logging
import time
import datetime
import json
import os
import numpy as np
import multiprocessing as mp
from src.model.nmf import NMF


logger = logging.getLogger("NMF")
logger.setLevel(logging.DEBUG)


class BatchNMF:
    def __init__(self,
                 V: np.ndarray,
                 U: np.ndarray,
                 factors: int,
                 models: int = 20,
                 method: str = "ls-nmf",
                 seed: int = 42,
                 init_method: str = "column_mean",
                 init_norm: bool = True,
                 fuzziness: float = 5.0,
                 max_iter: int = 20000,
                 converge_delta: float = 0.1,
                 converge_n: int = 100,
                 parallel: bool = True,
                 optimized: bool = True,
                 verbose: bool = True
                 ):

        self.factors = factors
        self.method = method

        self.V = V
        self.U = U

        self.seed = seed

        self.models = models
        self.max_iter = max_iter
        self.converge_delta = converge_delta
        self.converge_n = converge_n

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.init_method = init_method
        self.init_norm = init_norm
        self.fuzziness = fuzziness

        self.parallel = parallel
        self.optimized = optimized
        self.verbose = verbose

        self.results = []
        self.best_model = None

    def train(self):
        t0 = time.time()
        if self.parallel:
            # TODO: Add batch processing for large datasets and large number of epochs to reduce memory requirements.
            pool = mp.Pool()

            input_parameters = []
            for i in range(self.models):
                _seed = self.rng.integers(low=0, high=1e5)
                _nmf = NMF(
                    factors=self.factors,
                    method=self.method,
                    V=self.V,
                    U=self.U,
                    seed=_seed,
                    optimized=self.optimized
                )
                _nmf.initialize(init_method=self.init_method, init_norm=self.init_norm, fuzziness=self.fuzziness)
                input_parameters.append((_nmf, i))

            results = pool.starmap(self._train_task, input_parameters)

            best_epoch = -1
            best_q = float("inf")
            ordered_results = [None for i in range(len(results))]
            for result in results:
                epoch = int(result["epoch"])
                ordered_results[epoch] = result
                if result["Q"] < best_q:
                    best_q = result["Q"]
                    best_epoch = epoch
            if self.verbose:
                for result in ordered_results:
                    logger.info(f"Model: {result['epoch']}, Q: {round(result['Q'], 4)}, Seed: {result['seed']}, "
                                f"Converged: {result['converged']}, Steps: {result['steps']}/{self.max_iter}")
            self.results = ordered_results
            pool.close()
        else:
            self.results = []
            best_Q = float("inf")
            best_epoch = None
            for epoch in range(self.models):
                _seed = self.rng.integers(low=0, high=1e5)
                _nmf = NMF(
                    factors=self.factors,
                    method=self.method,
                    V=self.V,
                    U=self.U,
                    seed=_seed
                )
                _nmf.initialize(init_method=self.init_method, init_norm=self.init_norm, fuzziness=self.fuzziness)
                run = _nmf.train(max_iter=self.max_iter, converge_delta=self.converge_delta, converge_n=self.converge_n,
                                 epoch=epoch)
                if run == -1:
                    logger.error("Unable to execute batch run of NMF models.")
                    pass
                if _nmf.Qtrue < best_Q:
                    best_Q = _nmf.Qtrue
                    best_epoch = epoch
                self.results.append(
                    {
                        "epoch": epoch,
                        "Q": float(_nmf.Qtrue),
                        "steps": _nmf.converge_steps,
                        "converged": _nmf.converged,
                        "H": _nmf.H,
                        "W": _nmf.W,
                        "wh": _nmf.WH,
                        "seed": int(_seed)
                    }
                )
        t1 = time.time()
        logger.info(f"Results - Best Model: {best_epoch}, Q: {self.results[best_epoch]['Q']}, Converged: {self.results[best_epoch]['converged']}")
        logger.info(f"Runtime: {round((t1 - t0) / 60, 2)} min(s)")
        self.best_model = best_epoch

    def _train_task(self, nmf, epoch):
        t0 = time.time()
        nmf.train(max_iter=self.max_iter, converge_delta=self.converge_delta, converge_n=self.converge_n, epoch=epoch)
        t1 = time.time()
        if self.verbose:
            print(f"Model: {epoch}, Seed: {nmf.seed}, Q(true): {round(nmf.Qtrue, 4)}, "
                        f"Steps: {nmf.converge_steps}/{self.max_iter}, Converged: {nmf.converged}, "
                        f"Runtime: {round(t1 - t0, 2)} sec")
        return {
            "epoch": epoch,
            "Q": float(nmf.Qtrue),
            "steps": nmf.converge_steps,
            "converged": nmf.converged,
            "H": nmf.H,
            "W": nmf.W,
            "wh": nmf.WH,
            "seed": int(nmf.seed)
        }

    def save(self, output_name: str = None, output_path: str = None):
        if output_name is None:
            output_name = f"results_{datetime.datetime.now().strftime('%d-%m-%Y_%H%M%S')}.json"
        if output_path is None:
            output_path = "."
        elif not os.path.exists(output_path):
            os.mkdir(output_path)
        full_output_path = os.path.join(output_path, output_name)
        processed_results = []
        for result in self.results:
            processed_result = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    v = v.astype(float).tolist()
                processed_result[k] = v
            processed_results.append(processed_result)
        with open(full_output_path, 'w') as json_file:
            json.dump(processed_results, json_file)
            logger.info(f"Results saved to: {full_output_path}")


if __name__ == "__main__":

    import os
    from src.data.datahandler import DataHandler
    from src.utils import calculate_Q
    from tests.factor_comparison import FactorComp

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    t0 = time.time()
    for dataset in ["b", "sl", "br"]:           # "br", "sl", "b", "w"
        for method in ["ls-nmf"]:     # "ls-nmf", "ws-nmf"
            for factors in range(3, 13):
                # factors = 4
                # method = "ls-nmf"                   # "ls-nmf", "ws-nmf"
                init_method = "col_means"           # default is column means, "kmeans", "cmeans"
                init_norm = True
                seed = 40
                models = 20
                if method == "ws-nmf":
                    converge_delta = 1.0 if dataset == "b" else 0.1
                    max_iterations = 2000
                    converge_n = 5
                else:
                    max_iterations = 50000
                    converge_delta = 0.001
                    converge_n = 10
                parallel = True
                optimized = True
                # dataset = "br"          # "br": Baton Rouge, "b": Baltimore, "sl": St Louis
                index_col = "Date"

                if dataset == "br":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"br{factors}f_residuals.txt")
                elif dataset == "b":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_con.txt")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_unc.txt")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Baltimore")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"b{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"b{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"b{factors}f_residuals.txt")
                elif dataset == "sl":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-con.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-unc.csv")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "StLouis")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"sl{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"sl{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"sl{factors}f_residuals.txt")
                elif dataset == "w":
                    input_file = os.path.join("D:\\", "projects", "nmf_py", "user_data", "wash_con_cleaned.csv")
                    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "user_data", "wash_unc_cleaned.csv")
                    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Washington")
                    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"w{factors}f_profiles.txt")
                    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"w{factors}f_contributions.txt")
                    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                                      f"w{factors}f_residuals.txt")
                    index_col = "obs_date"

                sn_threshold = 2.0

                dh = DataHandler(
                    input_path=input_file,
                    uncertainty_path=uncertainty_file,
                    index_col=index_col,
                    sn_threshold=sn_threshold
                )
                V = dh.input_data_processed
                U = dh.uncertainty_data_processed

                batch_nmf = BatchNMF(V=V, U=U, factors=factors, models=models, method=method, seed=seed, init_method=init_method,
                                     init_norm=init_norm, fuzziness=5.0, max_iter=max_iterations, converge_delta=converge_delta,
                                     converge_n=converge_n, parallel=parallel, optimized=optimized)
                t0 = time.time()
                batch_nmf.train()

                t1 = time.time()
                # full_output_path = f"{dataset}-nmf-output-f{factors}.json"
                # batch_nmf.save(output_name=full_output_path)
                #
                # profile_comparison = FactorComp(nmf_output_file=full_output_path, pmf_profile_file=pmf_profile_file,
                #                                 pmf_contribution_file=pmf_contribution_file, factors=factors,
                #                                 species=len(dh.features), residuals_path=pmf_residuals_file)
                # pmf_q = calculate_Q(profile_comparison.pmf_residuals.values, dh.uncertainty_data_processed)
                # profile_comparison.compare(PMF_Q=pmf_q)
                # os.remove(path=full_output_path)
                runtime = round(t1-t0, 2)
                print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")

                run_key = f"{dataset}-{factors}"
                analysis_results = {
                    run_key:
                        {
                            "dataset": dataset,
                            "factors": factors,
                            f"{method}-runtime": runtime,
                            f"{method}-Q": float(batch_nmf.results[batch_nmf.best_epoch]["Q"])
                        }
                }
                current_results = {}
                analysis_file = "runtime_analysis.json"
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r') as json_file:
                        current_results = json.load(json_file)
                        if run_key in current_results.keys():
                            for k, v in analysis_results[run_key].items():
                                current_results[run_key].update({k: v})
                        else:
                            current_results[run_key] = analysis_results[run_key]
                else:
                    current_results = analysis_results
                with open(analysis_file, 'w') as json_file:
                    current_results = dict(sorted(current_results.items()))
                    json.dump(current_results, json_file)
                logger.info(f"Completed method: {method}, factors: {factors}, dataset: {dataset}")
