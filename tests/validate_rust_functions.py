import os
import time
import logging
import numpy as np
from src.model.base_nmf import BaseSearch, BaseNMF
from src.data.datahandler import DataHandler
from tests.factor_comparison import FactorComp
from src.utils import calculate_Q, q_loss
from nmf_pyr import nmf_pyr


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


if __name__ == "__main__":

    t0 = time.time()
    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col
    )

    n_components = 7
    method = "mu"                   # "kl", "ls-nmf", "is", "euc", "gd"
    V = dh.input_data_processed.astype(np.float64)
    U = dh.uncertainty_data_processed.astype(np.float64)
    seed = 42
    epochs = 10
    max_iterations = 20000
    converge_delta = 0.01
    converge_n = 100

    nmf = BaseNMF(n_components=n_components, V=V, U=U, seed=seed, method=method)

    UR = np.divide(1.0, U).astype(np.float64)
    test_u1 = nmf_pyr.py_matrix_reciprocal(U)
    test_result1 = UR - test_u1
    print(f"Matrix Reciprocal Test - Sum Difference: {np.sum(np.abs(test_result1))}")

    test_uv = np.multiply(UR, V)
    test_uv1 = nmf_pyr.py_matrix_multiply(UR, V)
    test_result2 = test_uv - test_uv1
    print(f"Matrix Multiply Test - Sum Difference: {np.sum(np.abs(test_result2))}")

    test_uv2 = np.multiply(V, V)
    test_uv3 = nmf_pyr.py_matrix_multiply(V, V)
    test_result2b = test_uv - test_uv1
    print(f"Matrix Multiply 2 Test - Sum Difference: {np.sum(np.abs(test_result2b))}")

    test_sub1 = np.subtract(V, U)
    test_sub2 = nmf_pyr.py_matrix_subtract(V, U)
    test_result_sub1 = test_sub1 - test_sub2
    print(f"Matrix Subtract Test - Sum Difference: {np.sum(np.abs(test_result_sub1))}")

    test_uvd = V / U
    test_uvd2 = np.divide(V, U)
    test_uvd1 = nmf_pyr.py_matrix_division(V, U)
    test_result3 = test_uvd - test_uvd1
    print(f"Matrix Divison Test - Sum Difference: {np.sum(np.abs(test_result3))}")

    test_sum1 = np.sum(V)
    test_sum2 = nmf_pyr.py_matrix_sum(V)
    test_result_sum = test_sum1 - test_sum2
    print(f"Matrix Sum Test - Difference: {np.sum(np.abs(test_result_sum))}")

    raw_test = np.zeros(V.shape)
    for (i, x) in enumerate(nmf.W):
        for (j, y) in enumerate(nmf.H.T):
            raw_test[i][j] = np.sum(x * y)

    test_wh = np.dot(nmf.W, nmf.H)
    test_wh2 = np.matmul(nmf.W, nmf.H)
    test_wh1 = nmf_pyr.py_matrix_mul(nmf.W, nmf.H)
    test_result4 = test_wh2 - test_wh1
    print(f"Matrix Mul Test - Sum Difference: {np.sum(np.abs(test_result4))}")

    test_q1 = nmf_pyr.py_calculate_q(V, U, nmf.W, nmf.H)
    test_q2 = q_loss(V=V, U=U, W=nmf.W, H=nmf.H)
    test_result5 = test_q1 - test_q2
    print(f"Calcualte Q Test - Sum Difference: {np.sum(np.abs(test_result5))}")

    _H = nmf.H
    _W = nmf.W
    nmf_results2 = nmf_pyr.nmf_kl(V, U, _W, _H, 1.0, 50, 0.1, 10)[0]
    W2, H2, q2, converged2, i_steps2, q_list = nmf_results2

    nmf.train(epoch=1, max_iterations=50, converge_delta=0.01, converge_n=10)
    W1 = nmf.W
    H1 = nmf.H
    q1 = q_loss(V=V, U=U, W=W1, H=H1)
    print(f"NMF KL 1 step - Q Difference: {round(np.abs(q1 - q2), 2)}, W Difference: {round(np.sum(np.abs(W1 - W2)), 2)}, H Difference: {round(np.sum(np.abs(H1 - H2)), 2)}")

    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")
