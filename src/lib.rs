extern crate core;

use rayon::prelude::*;
use std::collections::vec_deque::VecDeque;
use pyo3::{prelude::*, types::PyTuple};
use numpy::pyo3::Python;
use numpy::{PyReadonlyArrayDyn, ToPyArray, PyArrayDyn};
use nalgebra::{dmatrix, DMatrix, OMatrix, Dyn, Matrix};


/// NMF-PY Rust module
#[pymodule]
fn nmf_pyr(py: Python<'_>, m: &PyModule) -> PyResult<()> {

    /// NMF - Multiplicative-Update (Kullback-Leibler)
    /// Returns (W, H, Q, converged)
    #[pyfn(m)]
    fn nmf_kl<'py>(
        py: Python<'py>,
        v: PyReadonlyArrayDyn<'py, f64>, u: PyReadonlyArrayDyn<'py, f64>,
        w: PyReadonlyArrayDyn<'py, f64>, h: PyReadonlyArrayDyn<'py, f64>,
        update_weight: f64,
        max_i: i32, converge_delta: f64, converge_i: i32
    ) -> Result<&'py PyTuple, PyErr> {

        let v = OMatrix::<f64, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap().to_owned());
        let u = OMatrix::<f64, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap().to_owned());
        let mut new_w = OMatrix::<f64, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap().to_owned()).transpose();
        let mut new_h = OMatrix::<f64, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap().to_owned()).transpose();

        let mut q: f64 = calculate_q(&v, &u, &new_w, &new_h);
        let mut converged: bool = false;

        let uv = v.component_div(&u);
        let ur = matrix_reciprocal(&u);

        let mut best_q: f64 = q.clone();
        let mut best_h = new_h.clone();
        let mut best_w = new_w.clone();
        let mut q_list: VecDeque<f64> = VecDeque::new();

        let mut wh: DMatrix<f64>;
        let mut h1: DMatrix<f64>;
        let mut h2: DMatrix<f64>;
        let mut w1: DMatrix<f64>;
        let mut w2: DMatrix<f64>;
        let mut q_new = q.clone();
        let mut best_i = 0;

        let mut update_weight = update_weight.clone();

        let mut h_delta: DMatrix<f64>;
        let mut w_delta: DMatrix<f64>;

        for i in 0..max_i {
            wh = &new_w * &new_h;
            h1 = &new_w.transpose() * uv.component_div(&wh);
            h2 = matrix_reciprocal(&(new_w.transpose() * &ur));
            h_delta = update_weight * h2.component_mul(&h1);
            new_h = new_h.component_mul(&h_delta);

            wh = &new_w * &new_h;
            w1 = uv.component_div(&wh) * &new_h.transpose();
            w2 = matrix_reciprocal(&(&ur * &new_h.transpose()));
            w_delta = update_weight * w2.component_mul(&w1);
            new_w = new_w.component_mul(&w_delta);

            q_new = calculate_q(&v, &u, &new_w, &new_h);
            best_i = i;
            if q_new < best_q {
                best_q = q_new.clone();
                best_w = new_w.clone();
                best_h = new_h.clone();
            }
            q_list.push_back(q_new.clone());
            if (q_list.len() as i32) >= converge_i {
                let q_sum: f64 = q_list.iter().sum();
                let q_avg: f64 = q_sum / q_list.len() as f64;
                if (q_avg - q_new).abs() < converge_delta {
                    if update_weight < 0.01 {
                        converged = true;
                        break
                    }
                    else {
                        new_w = best_w.clone();
                        new_h = best_h.clone();
                        update_weight = &update_weight - 0.1;
                        q_list.clear();
                    }
                }
                else if (q_list.front().unwrap() - q_list.back().unwrap()) > 1.0 {
                    new_w = best_w.clone();
                    new_h = best_h.clone();
                    update_weight = &update_weight - 0.1;
                    q_list.clear();
                }
                q_list.pop_front();
            }
            q = q_new;
        }

        best_w = best_w.transpose();
        best_h = best_h.transpose();
        let w_matrix = Vec::from(best_w.data.as_vec().to_owned());
        let h_matrix = Vec::from(best_h.data.as_vec().to_owned());
        let final_w = w_matrix.to_pyarray(py).reshape(w.dims()).unwrap().to_dyn();
        let final_h = h_matrix.to_pyarray(py).reshape(h.dims()).unwrap().to_dyn();

        let values: PyObject = (final_w, final_h, best_q, converged, best_i, Vec::from(q_list)).into_py(py);
        let results: &PyTuple = PyTuple::new(py, values.downcast::<PyTuple>(py).iter());
        PyResult::Ok(results)
    }

    fn calculate_q(
        v: &OMatrix<f64, Dyn, Dyn>, u: &OMatrix<f64, Dyn, Dyn>,
        w: &OMatrix<f64, Dyn, Dyn>, h: &OMatrix<f64, Dyn, Dyn>
    ) -> f64 {
        let wh = w * h;
        let residuals = v - &wh;
        let weighted_residuals = residuals.component_div(&u).transpose();
        let wr2 = weighted_residuals.component_mul(&weighted_residuals);
        let q: f64 = wr2.sum();
        return q
    }

    fn matrix_reciprocal(m: &DMatrix<f64>) -> DMatrix<f64> {
        let vec_result: Vec<f64> = m.iter().map(|&i1| (1.0/i1)).collect();
        let result: DMatrix<f64> = DMatrix::from_vec(m.nrows(), m.ncols(), vec_result);
        return result
    }

    fn matrix_multiply(m1: &DMatrix<f64>, m2: &DMatrix<f64>) -> DMatrix<f64> {
        let vec_result = m1.iter().zip(m2.iter()).map(|(&i1, &i2)| (i1 * i2)).collect();
        let result: DMatrix<f64> = DMatrix::from_vec(m1.nrows(), m1.ncols(), vec_result);
        return result
    }

    fn matrix_subtract(m1: &DMatrix<f64>, m2: &DMatrix<f64>) -> DMatrix<f64> {
        let vec_result = m1.iter().zip(m2.iter()).map(|(&i1, &i2)| (i1 - i2)).collect();
        let result: DMatrix<f64> = DMatrix::from_vec(m1.nrows(), m1.ncols(), vec_result);
        return result
    }

    fn matrix_mul(m1: &DMatrix<f64>, m2: &DMatrix<f64>) -> DMatrix<f64> {
        let mat_result = m1 * m2;
        return mat_result
    }

    fn matrix_mul2(m1: &DMatrix<f64>, m2: &DMatrix<f64>) -> DMatrix<f64> {
        let matrix1 = m1.clone();
        let matrix2 = m2.clone();
        let mut matrix3 = DMatrix::<f64>::zeros(m1.shape().0, m2.shape().1);
        for (i, x) in matrix1.row_iter().enumerate() {
            for (j, y) in matrix2.column_iter().enumerate() {
                matrix3[(i, j)] = (x * y).sum();
            }
        }
        return matrix3
    }

    fn matrix_division(m1: &DMatrix<f64>, m2: &DMatrix<f64>) -> DMatrix<f64> {
        let div_results = m1.component_div(&m2);
        return div_results;
    }

    #[pyfn(m)]
    fn py_matrix_sum<'py>(py: Python<'py>, m: &'py PyArrayDyn<f64>) -> f64 {
        let mut new_matrix = OMatrix::<f64, Dyn, Dyn>::from_vec(m.dims()[0], m.dims()[1], m.to_vec().unwrap().to_owned()).transpose();
        let result = new_matrix.sum();
        return result
    }

    #[pyfn(m)]
    fn py_matrix_reciprocal<'py>(py: Python<'py>, m: &'py PyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {
        let mut new_matrix = OMatrix::<f64, Dyn, Dyn>::from_vec(m.dims()[0], m.dims()[1], m.to_vec().unwrap().to_owned()).transpose();
        new_matrix = matrix_reciprocal(&new_matrix);
        let result_matrix = Vec::from(new_matrix.data.as_vec().to_owned());
        let result = result_matrix.to_pyarray(py).reshape((m.dims()[0], m.dims()[1])).unwrap().to_dyn();
        return result
    }

    #[pyfn(m)]
    fn py_matrix_subtract<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f64>, m2: &'py PyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {
        let new_matrix1 = OMatrix::<f64, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap().to_owned()).transpose();
        let new_matrix2 = OMatrix::<f64, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap().to_owned()).transpose();
        let new_matrix = new_matrix1 - new_matrix2;
        let result_matrix = Vec::from(new_matrix.data.as_vec().to_owned());
        let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
        return result
    }

    #[pyfn(m)]
    fn py_matrix_multiply<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f64>, m2: &'py PyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {
        let new_matrix1 = OMatrix::<f64, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap().to_owned()).transpose();
        let new_matrix2 = OMatrix::<f64, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap().to_owned()).transpose();
        let new_matrix = new_matrix1.component_mul(&new_matrix2);
        let result_matrix = Vec::from(new_matrix.data.as_vec().to_owned());
        let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
        return result
    }

    #[pyfn(m)]
    fn py_matrix_mul<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f64>, m2: &'py PyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {
        let matrix1 = OMatrix::<f64, Dyn, Dyn>::from_row_iterator(m1.dims()[0], m1.dims()[1], m1.to_owned_array().into_iter());
        let matrix2 = OMatrix::<f64, Dyn, Dyn>::from_row_iterator(m2.dims()[0], m2.dims()[1], m2.to_owned_array().into_iter());
        let new_matrix = matrix_mul(&matrix1, &matrix2).transpose();
        let result_matrix = Vec::from(new_matrix.data.as_vec().to_owned());
        let result = result_matrix.to_pyarray(py).reshape((m1.dims()[0], m2.dims()[1])).unwrap().to_dyn();
        return result
    }

    #[pyfn(m)]
    fn py_matrix_division<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f64>, m2: &'py PyArrayDyn<f64>) -> &'py PyArrayDyn<f64> {
        let matrix1 = OMatrix::<f64, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap().to_owned()).transpose();
        let matrix2 = OMatrix::<f64, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap().to_owned()).transpose();
        let new_matrix = matrix_division(&matrix1, &matrix2);
        let result_matrix = Vec::from(new_matrix.data.as_vec().to_owned());
        let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
        return result
    }

    #[pyfn(m)]
    fn py_calculate_q<'py>(py: Python<'py>,
                           v: &'py PyArrayDyn<f64>, u: &'py PyArrayDyn<f64>,
                           w: &'py PyArrayDyn<f64>, h: &'py PyArrayDyn<f64>) -> f64 {
        let matrix_v = OMatrix::<f64, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap().to_owned());
        let matrix_u = OMatrix::<f64, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap().to_owned());
        let matrix_w = OMatrix::<f64, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap().to_owned()).transpose();
        let matrix_h = OMatrix::<f64, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap().to_owned()).transpose();
        let q = calculate_q(&matrix_v, &matrix_u, &matrix_w, &matrix_h);
        return q
    }
    Ok(())
}