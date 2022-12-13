from src.model.model import NMFModel
from src.data.datahandler import DataHandler
from tests.factor_comparison import FactorComp
import os
import time


if __name__ == "__main__":

    t0 = time.time()

    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")

    quantile = 0.95
    drop_max = True
    drop_min = False

    epochs = 10
    n_components = 4
    max_iterations = 20000
    seed = 42
    use_original_convergence = False

    lr_initial = 1e-1
    lr_decay_steps = 250
    lr_decay_rate = 0.95
    converge_diff = 0.01
    converge_iter = 50

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col
    )
    # dh.remove_outliers(quantile=quantile, drop_min=drop_min, drop_max=drop_max)

    model = NMFModel(
        dh=dh,
        epochs=epochs,
        n_components=n_components,
        max_iterations=max_iterations,
        seed=seed,
        lr_initial=lr_initial,
        lr_decay_steps=lr_decay_steps,
        lr_decay_rate=lr_decay_rate,
        converge_diff=converge_diff,
        converge_iter=converge_iter,
        use_original_convergence=use_original_convergence,
    )
    model.fit()
    model.print_results()

    full_output_path = "test-save-04.json"
    model.save(output_name=full_output_path)
    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 3)} min(s)")

    pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", "baton-rouge_4f_profiles.txt")
    profile_comparison = FactorComp(nmf_output=full_output_path, pmf_output=pmf_file, factors=4, species=41)
    profile_comparison.compare()
