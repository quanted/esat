import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "..\\src")

import copy
import time
import logging
import numpy as np
import pandas as pd
from  esat.model.sa import SA
from  esat.data.datahandler import DataHandler
from  esat.metrics import calculate_Q, q_loss, qr_loss
from esat import esat_rust

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


if __name__ == "__main__":

    t0 = time.time()
    input_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "esat", "data", "Dataset-BatonRouge-unc.csv")

    index_col = "Date"

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        index_col=index_col
    )

    n_components = 7
    method = "ls-nmf"
    V = dh.input_data_processed.astype(np.float64)
    U = dh.uncertainty_data_processed.astype(np.float64)
    seed = 42
    epochs = 10
    max_iterations = 20000
    converge_delta = 0.01
    converge_n = 100

    sa = SA(factors=n_components, V=V, U=U, seed=seed, method=method)
    sa.initialize()

    UR = np.divide(1.0, U).astype(np.float64)
    test_u1 = esat_rust.py_matrix_reciprocal(U)
    test_result1 = UR - test_u1
    print(f"Matrix Reciprocal Test - Sum Difference: {np.sum(np.abs(test_result1))}")

    test_uv = np.multiply(UR, V)
    test_uv1 = esat_rust.py_matrix_multiply(UR, V)
    test_result2 = test_uv - test_uv1
    print(f"Matrix Multiply Test - Sum Difference: {np.sum(np.abs(test_result2))}")

    test_uv2 = np.multiply(V, V)
    test_uv3 = esat_rust.py_matrix_multiply(V, V)
    test_result2b = test_uv - test_uv1
    print(f"Matrix Multiply 2 Test - Sum Difference: {np.sum(np.abs(test_result2b))}")

    test_sub1 = np.subtract(V, U)
    test_sub2 = esat_rust.py_matrix_subtract(V, U)
    test_result_sub1 = test_sub1 - test_sub2
    print(f"Matrix Subtract Test - Sum Difference: {np.sum(np.abs(test_result_sub1))}")

    test_uvd = V / U
    test_uvd2 = np.divide(V, U)
    test_uvd1 = esat_rust.py_matrix_division(V, U)
    test_result3 = test_uvd - test_uvd1
    print(f"Matrix Divison Test - Sum Difference: {np.sum(np.abs(test_result3))}")

    test_sum1 = np.sum(V)
    test_sum2 = esat_rust.py_matrix_sum(V)
    test_result_sum = test_sum1 - test_sum2
    print(f"Matrix Sum Test - Difference: {np.sum(np.abs(test_result_sum))}")

    raw_test = np.zeros(V.shape)
    for (i, x) in enumerate(sa.W):
        for (j, y) in enumerate(sa.H.T):
            raw_test[i][j] = np.sum(x * y)

    test_wh = np.dot(sa.W, sa.H)
    test_wh2 = np.matmul(sa.W, sa.H)
    test_wh1 = esat_rust.py_matrix_mul(sa.W, sa.H)
    test_result4 = test_wh2 - test_wh1
    print(f"Matrix Mul Test - Sum Difference: {np.sum(np.abs(test_result4))}")

    test_q1 = esat_rust.py_calculate_q(V, U, sa.W, sa.H)
    test_q2 = q_loss(V=V, U=U, W=sa.W, H=sa.H)
    test_result5 = test_q1 - test_q2
    print(f"Calcualte Q Test - Sum Difference: {np.sum(np.abs(test_result5))}")

    test_qr1, un1 = esat_rust.py_calculate_q_robust(V, U, sa.W, sa.H, 4.0)
    test_qr2, un2 = qr_loss(V=V, U=U, W=sa.W, H=sa.H, alpha=4.0)
    test_result6 = test_qr1 - test_qr2
    test_results7 = un1 - un2
    print(f"Calcualte Q(robust) Test - Sum Difference: {np.sum(np.abs(test_result6))}, "
          f"Uncertainty difference: {np.sum(test_results7)}")

    _index = list(range(0, V.shape[0]))
    _H = sa.H
    _W = sa.W
    _We = sa.We
    _V = V[_index].transpose().reshape(V.shape)
    v0 = V - _V
    _U = U[_index].transpose().reshape(V.shape)
    nmf_results2 = esat_rust.ls_nmf(_V, _U, _We, _W, _H, 50, 0.1, 10, False, 100, 4)[0]
    W2, H2, q2, converged2, i_steps2, q_list = nmf_results2

    sa.train(max_iter=50, converge_delta=0.01, converge_n=10)
    W1 = sa.W
    H1 = sa.H
    q1 = q_loss(V=V, U=U, W=W1, H=H1)
    print(f"NMF KL 1 step - Q Difference: {round(np.abs(q1 - q2), 2)}, W Difference: {round(np.sum(np.abs(W1 - W2)), 2)}, H Difference: {round(np.sum(np.abs(H1 - H2)), 2)}")

    max_iter = 2000
    for n_factors in range(3, 8):
        for method in ('ls-nmf', 'ws-nmf'):
            converge_delta = 0.01 if method == 'ls-nmf' else 0.1
            converge_n = 20 if method == 'ls-nmf' else 10
            _index = list(range(0, V.shape[0]))
            _V = copy.deepcopy(V[_index])
            _U = copy.deepcopy(U[_index])
            for i in _index:
                for j in range(0, V.shape[1]):
                    V[i,j] = _V[i, j]
                    U[i,j] = _U[i, j]
            ta0 = time.time()
            sa_py = SA(factors=n_factors, V=V, U=U, seed=seed, method=method, optimized=False)
            sa_py.initialize()
            sa_py.train(max_iter=max_iter, converge_delta=converge_delta, converge_n=converge_n)
            ta1 = time.time()
            # _index = list(range(0, V.shape[0]))
            _V = copy.deepcopy(V[_index])
            _U = copy.deepcopy(U[_index])
            for i in _index:
                for j in range(0, V.shape[1]):
                    V[i, j] = _V[i, j]
                    U[i, j] = _U[i, j]
            tb0 = time.time()
            sa_r = SA(factors=n_factors, V=V, U=U, seed=seed, method=method, optimized=True, parallelized=True)
            sa_r.initialize()
            sa_r.train(max_iter=max_iter, converge_delta=converge_delta, converge_n=converge_n)
            tb1 = time.time()
            # print(f"{method} {n_factors} - Python Q: {sa_py.Qtrue}, Rust Q: {sa_r.Qtrue}")
            print(f"{method} {n_factors} - Python runtime: {round((ta1-ta0), 2)} secs, Rust runtime: {round((tb1-tb0), 2)} secs")
