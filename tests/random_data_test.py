from src.model.base_nmf import BaseSearch
from src.data.datahandler import DataHandler
import os
import time


if __name__ == "__main__":

    t0 = time.time()

    name = "test-sample-03"

    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "samples", f"{name}-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "samples", f"{name}-unc.csv")
    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "samples")

    sample_count = 200000
    species_count = 35
    value_min = 0.001
    value_max = 10.0

    epochs = 2
    n_components = 4
    max_iterations = 20000
    seed = 42

    converge_diff = 0.5
    converge_iter = 10

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col,
        generate_data=True
    )
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed

    bs = BaseSearch(n_components=n_components, method="mu", V=V, U=U, seed=seed, epochs=epochs, max_iterations=max_iterations,
                    converge_delta=0.1, converge_n=100)
    # bs.train()
    bs.parallel_train()

    full_output_path = f"{name}-results.json"
    bs.save(output_name=full_output_path)
    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 3)} min(s)")
