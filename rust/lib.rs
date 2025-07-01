extern crate core;

use std::collections::vec_deque::VecDeque;
use pyo3::{prelude::*, types::PyTuple};
use numpy::pyo3::Python;
use numpy::{PyReadonlyArrayDyn, ToPyArray, PyArrayDyn};
use nalgebra::*;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget};
use std::io::{self, Write};
use console::Term;


/// ESAT Rust module
#[pymodule]
fn esat_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    const EPSILON: f32 = 1e-12;

    #[pyfn(m)]
    fn clear_screen() -> PyResult<()> {
        print!("\x1B[2J\x1B[H");
        io::stdout().flush().unwrap_or(());
        Ok(())
    }

    // NMF - Least-Squares Algorithm (LS-NMF)
    // Returns (W, H, Q, converged)
    #[pyfn(m)]
    fn ls_nmf<'py>(
        py: Python<'py>,
        v: PyReadonlyArrayDyn<'py, f32>,
        u: PyReadonlyArrayDyn<'py, f32>,
        we: PyReadonlyArrayDyn<'py, f32>,
        w: PyReadonlyArrayDyn<'py, f32>,
        h: PyReadonlyArrayDyn<'py, f32>,
        max_iter: i32,
        converge_delta: f32,
        converge_n: i32,
        robust_alpha: f32,
        model_i: i8,
        static_h: Option<bool>,
        delay_h: Option<i32>,
    ) -> Result<&'py PyTuple, PyErr> {

        let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap().to_owned());
        let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap().to_owned());
        let we = OMatrix::<f32, Dyn, Dyn>::from_vec(we.dims()[0], we.dims()[1], we.to_vec().unwrap().to_owned());

        let mut new_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap().to_owned()).transpose();
        let mut new_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap().to_owned()).transpose();

        let mut new_we = we.clone();
        let mut q = 0.0;
        let mut qtrue: f32 = calculate_q(&v, &u, &new_w, &new_h);
        let mut qrobust: f32 = 0.0;

        let mut converged: bool = false;
        let mut converge_i: i32 = 0;
        let mut q_list: VecDeque<f32> = VecDeque::new();
        let mut q_list_full: VecDeque<f32> = VecDeque::new();
        let mut mse: f32 = 0.0;
        let datapoints = v.len() as f32;

        let mut wh: DMatrix<f32>;
        let mut h_num: DMatrix<f32>;
        let mut h_den: DMatrix<f32>;
        let mut w_num: DMatrix<f32>;
        let mut w_den: DMatrix<f32>;
        let mut wev: DMatrix<f32> = new_we.component_mul(&v);
        let hold_h = static_h.unwrap_or(false);
        let delay_h = delay_h.unwrap_or(-1);

        let mut robust_results = calculate_q_robust(&v, &u, &new_w, &new_h, robust_alpha);

        let location_i = (model_i as usize).try_into().unwrap();
        let term = Term::buffered_stdout();
        term.move_cursor_to(0, location_i).unwrap();
        let draw_target = ProgressDrawTarget::term(term.clone(), 20);
        let pb = ProgressBar::with_draw_target(Some(max_iter as u64), draw_target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {per_sec} iter/sec - {msg}")
                .unwrap()
                .progress_chars("-|-"),
        );

        for i in 0..max_iter{
            if !hold_h || (delay_h > 0 && i > delay_h) {
                wh = &new_w * &new_h;
                h_num = new_w.transpose() * &wev;
                h_den = &new_w.transpose() * &new_we.component_mul(&wh);
                new_h = new_h.component_mul(&h_num.component_div(&h_den));
            }
            wh = &new_w * &new_h;
            w_num = &wev * new_h.transpose();
            w_den = &new_we.component_mul(&wh) * &new_h.transpose();
            new_w = new_w.component_mul(&w_num.component_div(&w_den));

            qtrue = calculate_q(&v, &u, &new_w, &new_h);
            robust_results = calculate_q_robust(&v, &u, &new_w, &new_h, robust_alpha);
            qrobust = robust_results.0;
            q = qtrue;
            mse = qtrue / datapoints;
            q_list.push_back(q);
            q_list_full.push_back(q);
            converge_i = i;

            term.move_cursor_to(0, location_i).unwrap();
            pb.set_position((i+1) as u64);
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_message(format!("Model: {}, Q(True): {:.4}, Q(Robust): {:.4}, MSE: {:.4}", model_i, qtrue, qrobust, mse));

            if (q_list.len() as i32) >= converge_n {
                if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                        converged = true;
                        break
                }
                q_list.pop_front();
            }
        }
        term.move_cursor_to(0, location_i).unwrap();
        pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, Q(Robust): {:.4}, MSE: {:.4}", model_i, qtrue, qrobust, mse));

        new_w = new_w.transpose();
        new_h = new_h.transpose();
        let w_matrix = Vec::from(new_w.data.as_vec().to_owned());
        let h_matrix = Vec::from(new_h.data.as_vec().to_owned());
        let final_w = w_matrix.to_pyarray(py).reshape(w.dims()).unwrap().to_dyn();
        let final_h = h_matrix.to_pyarray(py).reshape(h.dims()).unwrap().to_dyn();

        let values: PyObject = (final_w, final_h, q, converged, converge_i, Vec::from(q_list_full)).into_py(py);
        let results: &PyTuple = PyTuple::new(py, values.downcast::<PyTuple>(py).iter());
        PyResult::Ok(results)
    }

    // NMF - Weight Semi-NMF algorithm
    // Returns (W, H, Q, converged)
    #[pyfn(m)]
    fn ws_nmf<'py>(
        py: Python<'py>,
        v: PyReadonlyArrayDyn<'py, f32>,
        u: PyReadonlyArrayDyn<'py, f32>,
        we: PyReadonlyArrayDyn<'py, f32>,
        w: PyReadonlyArrayDyn<'py, f32>,
        h: PyReadonlyArrayDyn<'py, f32>,
        max_iter: i32,
        converge_delta: f32,
        converge_n: i32,
        robust_alpha: f32,
        model_i: i8,
        static_h: Option<bool>,
        delay_h: Option<i32>,
    ) -> Result<&'py PyTuple, PyErr> {
        let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap());
        let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap());
        let we = OMatrix::<f32, Dyn, Dyn>::from_vec(we.dims()[0], we.dims()[1], we.to_vec().unwrap());

        let mut new_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap()).transpose();
        let mut new_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap()).transpose();
        let mut new_we = we.clone();

        let mut qtrue: f32 = calculate_q(&v, &u, &new_w, &new_h);
        let mut qrobust: f32 = 0.0;
        let mut q = qtrue.clone();

        let mut mse: f32 = 0.0;
        let datapoints = v.len() as f32;
        let hold_h = static_h.unwrap_or(false);
        let delay_h = delay_h.unwrap_or(-1);

        let mut converged: bool = false;
        let mut converge_i: i32 = 0;
        let mut q_list: VecDeque<f32> = VecDeque::new();
        let mut q_list_full: VecDeque<f32> = VecDeque::new();

        let mut we_j_diag: DMatrix<f32>;
        let mut v_j;

        let wev = &new_we.component_mul(&v);

        let location_i = (model_i as usize).try_into().unwrap();
        let term = Term::buffered_stdout();
        term.move_cursor_to(0, location_i).unwrap();
        let draw_target = ProgressDrawTarget::term(term.clone(), 20);
        let pb = ProgressBar::with_draw_target(Some(max_iter as u64), draw_target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {per_sec} iter/sec - {msg}")
                .unwrap()
                .progress_chars("-|-"),
        );

        for i in 0..max_iter{
            for (j, we_j) in new_we.row_iter().enumerate(){
                we_j_diag = DMatrix::from_diagonal(&DVector::from_row_slice(we_j.transpose().as_slice()));
                v_j = DVector::from_row_slice(wev.row(j).transpose().as_slice());

                let w_n = (&new_h * v_j).transpose();
                let uh = &we_j_diag * &new_h.transpose();
                let w_d = &new_h * &uh;
                let w_d_det = w_d.determinant();
                let w_d_inv: DMatrix<f32> = if w_d_det == 0.0 {
                    w_d.pseudo_inverse(1e-12).unwrap()
                }
                else{
                    w_d.try_inverse().unwrap()
                };
                let _w = w_n * w_d_inv;
                let _w_row = DVector::from_column_slice(_w.as_slice());
                let w_row = &_w_row.column(0).transpose();
                new_w.set_row(j, &w_row);
            }
            if !hold_h || (delay_h > 0 && i > delay_h) {
                let w_neg = (Matrix::abs(&new_w) - &new_w) / 2.0;
                let w_pos = (Matrix::abs(&new_w) + &new_w) / 2.0;
                for (j, we_j) in new_we.column_iter().enumerate(){
                    we_j_diag = DMatrix::from_diagonal(&DVector::from_column_slice(we_j.as_slice()));
                    v_j = DVector::from_row_slice(wev.column(j).as_slice());

                    let n1 = v_j.transpose() * &w_pos;
                    let d1 = v_j.transpose() * &w_neg;

                    let n2a = w_neg.transpose() * &we_j_diag;
                    let n2b = &n2a * &w_neg;
                    let d2a = w_pos.transpose() * &we_j_diag;
                    let d2b = &d2a * &w_pos;

                    let h_j = new_h.column(j).transpose();
                    let n2 = &h_j * &n2b;
                    let d2 = &h_j * &d2b;
                    let _n = (n1 + n2).add_scalar(EPSILON);
                    let _d = (d1 + d2).add_scalar(EPSILON);
                    let mut h_delta = _n.component_div(&_d);
                    h_delta = h_delta.map(|x| x.sqrt());
                    let _h = h_j.component_mul(&h_delta);
                    let h_row = DVector::from_row_slice(_h.as_slice());
                    new_h.set_column(j, &h_row);
                }
            }
            qtrue = calculate_q(&v, &u, &new_w, &new_h);
            let robust_results = calculate_q_robust(&v, &u, &new_w, &new_h, robust_alpha);
            qrobust = robust_results.0;
            q = qtrue;
            mse = qtrue / datapoints;

            term.move_cursor_to(0, location_i).unwrap();
            pb.set_position((i+1) as u64);
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_message(format!("Model: {}, Q(True): {:.4}, Q(Robust): {:.4}, MSE: {:.4}", model_i, qtrue, qrobust, mse));

            q_list.push_back(q);
            q_list_full.push_back(q);
            converge_i = i;
            if (q_list.len() as i32) >= converge_n {
                if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                        converged = true;
                        break
                }
                q_list.pop_front();
            }
        }
        term.move_cursor_to(0, location_i).unwrap();
        pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, Q(Robust): {:.4}, MSE: {:.4}", model_i, qtrue, qrobust, mse));

        new_w = new_w.transpose();
        new_h = new_h.transpose();
        let w_matrix = new_w.data.as_vec().to_owned();
        let h_matrix = new_h.data.as_vec().to_owned();
        let final_w = w_matrix.to_pyarray(py).reshape(w.dims()).unwrap().to_dyn();
        let final_h = h_matrix.to_pyarray(py).reshape(h.dims()).unwrap().to_dyn();

        let values: PyObject = (final_w, final_h, q, converged, converge_i, Vec::from(q_list_full)).into_py(py);
        let results: &PyTuple = PyTuple::new(py, values.downcast::<PyTuple>(py).iter());
        PyResult::Ok(results)
    }

    // NMF - Weight Semi-NMF algorithm
    // Returns (W, H, Q, converged)
    #[pyfn(m)]
    fn ws_nmf_p<'py>(
        py: Python<'py>,
        v: PyReadonlyArrayDyn<'py, f32>,
        u: PyReadonlyArrayDyn<'py, f32>,
        we: PyReadonlyArrayDyn<'py, f32>,
        w: PyReadonlyArrayDyn<'py, f32>,
        h: PyReadonlyArrayDyn<'py, f32>,
        max_iter: i32,
        converge_delta: f32,
        converge_n: i32,
        robust_alpha: f32,
        model_i: i8,
        static_h: Option<bool>,
        delay_h: Option<i32>,
    ) -> Result<&'py PyTuple, PyErr> {
        let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap());
        let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap());
        let we = OMatrix::<f32, Dyn, Dyn>::from_vec(we.dims()[0], we.dims()[1], we.to_vec().unwrap());

        let mut new_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap()).transpose();
        let mut new_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap()).transpose();
        let mut new_we = we.clone();

        let mut qtrue: f32 = calculate_q(&v, &u, &new_w, &new_h);
        let mut qrobust: f32 = 0.0;
        let mut q = qtrue.clone();
        let mut mse: f32 = 0.0;

        let mut converged: bool = false;
        let mut converge_i: i32 = 0;
        let mut q_list: VecDeque<f32> = VecDeque::new();
        let mut q_list_full: VecDeque<f32> = VecDeque::new();

        let m = v.nrows();
        let n = v.ncols();
        let mut w_prime: OMatrix<f32, Dyn, Dyn> = new_w.clone().transpose();
        let mut h_prime: OMatrix<f32, Dyn, Dyn> = new_h.clone();
        let datapoints = v.len() as f32;
        let hold_h = static_h.unwrap_or(false);
        let delay_h = delay_h.unwrap_or(-1);

        let location_i = (model_i as usize).try_into().unwrap();
        let term = Term::buffered_stdout();
        term.move_cursor_to(0, location_i).unwrap();
        let draw_target = ProgressDrawTarget::term(term.clone(), 20);
        let pb = ProgressBar::with_draw_target(Some(max_iter as u64), draw_target);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {per_sec} iter/sec - {msg}")
                .unwrap()
                .progress_chars("-|-"),
        );

        for i in 0..max_iter{
            let wev = &new_we.component_mul(&v);
            w_prime.par_column_iter_mut().zip(0..m).for_each(|(mut w_col, j)| {
                let we_j_diag = DMatrix::from_diagonal(&DVector::from_row_slice(new_we.row(j).transpose().as_slice()));
                let v_j = DVector::from_row_slice(wev.row(j).transpose().as_slice());

                let w_n = (&new_h * v_j).transpose();
                let uh = &we_j_diag * &new_h.transpose();
                let w_d = &new_h * &uh;
                let mut w_d_inv = w_d;
                if ! w_d_inv.try_inverse_mut() {
                    w_d_inv = w_d_inv.pseudo_inverse(1e-12).unwrap()
                }
                let _w = w_n * w_d_inv;
                let _j_w_row = DVector::from_column_slice(_w.as_slice());
                let j_w_row = _j_w_row.column(0);
                for k in 0..w_col.shape().0{
                    w_col[k] = j_w_row[k];
                }
            });
            new_w = w_prime.transpose();
            if !hold_h || (delay_h > 0 && i > delay_h) {
                let w_neg = (Matrix::abs(&new_w) - &new_w) / 2.0;
                let w_pos = (Matrix::abs(&new_w) + &new_w) / 2.0;

                h_prime.par_column_iter_mut().zip(0..n).for_each(|(mut h_col, j)| {
                    let we_j_diag = DMatrix::from_diagonal(&DVector::from_column_slice(new_we.column(j).as_slice()));
                    let v_j = DVector::from_row_slice(wev.column(j).as_slice());

                    let n1 = v_j.transpose() * &w_pos;
                    let d1 = v_j.transpose() * &w_neg;

                    let n2a = w_neg.transpose() * &we_j_diag;
                    let n2b = &n2a * &w_neg;
                    let d2a = w_pos.transpose() * &we_j_diag;
                    let d2b = &d2a * &w_pos;

                    let h_j = new_h.column(j).transpose();
                    let n2 = &h_j * &n2b;
                    let d2 = &h_j * &d2b;
                    let _n = (n1 + n2).add_scalar(EPSILON);
                    let _d = (d1 + d2).add_scalar(EPSILON);
                    let mut h_delta = _n.component_div(&_d);
                    h_delta = h_delta.map(|x| x.sqrt());
                    let _h = h_j.component_mul(&h_delta);
                    let j_h_row = DVector::from_column_slice(_h.as_slice());
                    for k in 0..h_col.shape().0{
                        h_col[k] = j_h_row[k];
                    }
                });
                new_h = h_prime.to_owned();
            }
            qtrue = calculate_q(&v, &u, &new_w, &new_h);
            let robust_results = calculate_q_robust(&v, &u, &new_w, &new_h, robust_alpha);
            qrobust = robust_results.0;
            mse = qtrue / datapoints;

            term.move_cursor_to(0, location_i).unwrap();
            pb.set_position((i+1) as u64);
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_message(format!("Model: {}, Q(True): {:.4}, Q(Robust): {:.4}, MSE: {:.4}", model_i, qtrue, qrobust, mse));

            q = qtrue.clone();
            q_list.push_back(q);
            q_list_full.push_back(q);
            converge_i = i;
            if (q_list.len() as i32) >= converge_n {
                if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                        converged = true;
                        break
                }
                q_list.pop_front();
            }
        }
        term.move_cursor_to(0, location_i).unwrap();
        pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, Q(Robust): {:.4}, MSE: {:.4}", model_i, qtrue, qrobust, mse));

        new_w = new_w.transpose();
        new_h = new_h.transpose();
        let w_matrix = new_w.data.as_vec().to_owned();
        let h_matrix = new_h.data.as_vec().to_owned();
        let final_w = w_matrix.to_pyarray(py).reshape(w.dims()).unwrap().to_dyn();
        let final_h = h_matrix.to_pyarray(py).reshape(h.dims()).unwrap().to_dyn();

        let values: PyObject = (final_w, final_h, q, converged, converge_i, Vec::from(q_list_full)).into_py(py);
        let results: &PyTuple = PyTuple::new(py, values.downcast::<PyTuple>(py).iter());
        PyResult::Ok(results)
    }

    /// NMF - Multiplicative-Update (Kullback-Leibler)
    /// Returns (W, H, Q, converged)
    #[pyfn(m)]
    fn nmf_kl<'py>(
        py: Python<'py>,
        v: PyReadonlyArrayDyn<'py, f32>, u: PyReadonlyArrayDyn<'py, f32>,
        w: PyReadonlyArrayDyn<'py, f32>, h: PyReadonlyArrayDyn<'py, f32>,
        update_weight: f32,
        max_i: i32, converge_delta: f32, converge_i: i32
    ) -> Result<&'py PyTuple, PyErr> {

        let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap());
        let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap());
        let mut new_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap()).transpose();
        let mut new_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap()).transpose();

        let mut q: f32 = calculate_q(&v, &u, &new_w, &new_h);
        let mut converged: bool = false;

        let uv = v.component_div(&u);
        let ur = matrix_reciprocal(&u);

        let mut best_q: f32 = q;
        let mut best_h = new_h.clone();
        let mut best_w = new_w.clone();
        let mut q_list: VecDeque<f32> = VecDeque::new();

        let mut wh: DMatrix<f32>;
        let mut h1: DMatrix<f32>;
        let mut h2: DMatrix<f32>;
        let mut w1: DMatrix<f32>;
        let mut w2: DMatrix<f32>;
        let mut best_i = 0;

        let mut update_weight = update_weight;

        let mut h_delta: DMatrix<f32>;
        let mut w_delta: DMatrix<f32>;

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

            q = calculate_q(&v, &u, &new_w, &new_h);
            best_i = i;
            if q < best_q {
                best_q = q;
                best_w = new_w.clone();
                best_h = new_h.clone();
            }
            q_list.push_back(q);
            if (q_list.len() as i32) >= converge_i {
                let q_sum: f32 = q_list.iter().sum();
                let q_avg: f32 = q_sum / q_list.len() as f32;
                if (q_avg - q).abs() < converge_delta {
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
        }

        best_w = best_w.transpose();
        best_h = best_h.transpose();
        let w_matrix = best_w.data.as_vec().to_owned();
        let h_matrix = best_h.data.as_vec().to_owned();
        let final_w = w_matrix.to_pyarray(py).reshape(w.dims()).unwrap().to_dyn();
        let final_h = h_matrix.to_pyarray(py).reshape(h.dims()).unwrap().to_dyn();

        let values: PyObject = (final_w, final_h, best_q, converged, best_i, Vec::from(q_list)).into_py(py);
        let results: &PyTuple = PyTuple::new(py, values.downcast::<PyTuple>(py).iter());
        PyResult::Ok(results)
    }

    fn calculate_q(
        v: &OMatrix<f32, Dyn, Dyn>, u: &OMatrix<f32, Dyn, Dyn>,
        w: &OMatrix<f32, Dyn, Dyn>, h: &OMatrix<f32, Dyn, Dyn>
    ) -> f32 {
        let wh = w * h;
        let residuals = v - &wh;
        let weighted_residuals = residuals.component_div(u).transpose();
        let wr2 = weighted_residuals.component_mul(&weighted_residuals);
        let q: f32 = wr2.sum();
        q
    }

    fn calculate_q_robust(
        v: &OMatrix<f32, Dyn, Dyn>, u: &OMatrix<f32, Dyn, Dyn>,
        w: &OMatrix<f32, Dyn, Dyn>, h: &OMatrix<f32, Dyn, Dyn>,
        robust_alpha: f32
    ) -> (f32, DMatrix<f32>) {
        let wh = w * h;
        let residuals = v - &wh;
        let scaled_residuals = residuals.component_div(u).abs();
        let robust_uncertainty = (&scaled_residuals / robust_alpha).map(|x| x.sqrt()).component_mul(&u);
        let robust_residuals = (residuals.component_div(&robust_uncertainty)).abs();
        let new_scaled_residuals = &scaled_residuals.clone();
        let merged_results = matrix_merge(&new_scaled_residuals, &robust_residuals, &u, &robust_uncertainty, robust_alpha);
        let merged_residuals = merged_results.0;
        let updated_uncertainty = merged_results.1;
        let mr2 = merged_residuals.component_mul(&merged_residuals);
        let q_robust = mr2.sum();
        (q_robust, updated_uncertainty)
    }

    fn matrix_merge(
        matrix1: &OMatrix<f32, Dyn, Dyn>,
        matrix2: &OMatrix<f32, Dyn, Dyn>,
        matrix4: &OMatrix<f32, Dyn, Dyn>,
        matrix5: &OMatrix<f32, Dyn, Dyn>,
        alpha: f32
    ) -> (DMatrix<f32>, DMatrix<f32>) {
        let mut matrix3 = DMatrix::<f32>::zeros(matrix1.shape().0, matrix1.shape().1);
        let mut matrix6 = DMatrix::<f32>::zeros(matrix4.shape().0, matrix4.shape().1);
        for (j, col) in matrix1.column_iter().enumerate(){
            for (i, row) in matrix1.row_iter().enumerate(){
                if matrix1[(i,j)] > alpha {
                    matrix3[(i,j)] = matrix2[(i,j)];
                    matrix6[(i,j)] = matrix5[(i,j)];
                }
                else{
                    matrix3[(i,j)] = matrix1[(i,j)];
                    matrix6[(i,j)] = matrix4[(i,j)];
                }
            }
        }
        (matrix3, matrix6)
    }

    fn matrix_reciprocal(m: &DMatrix<f32>) -> DMatrix<f32> {
        let vec_result: Vec<f32> = m.iter().map(|&i1| (1.0/i1)).collect();
        let result: DMatrix<f32> = DMatrix::from_vec(m.nrows(), m.ncols(), vec_result);
        result
    }

    fn matrix_multiply(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
        let vec_result = m1.iter().zip(m2.iter()).map(|(&i1, &i2)| (i1 * i2)).collect();
        let result: DMatrix<f32> = DMatrix::from_vec(m1.nrows(), m1.ncols(), vec_result);
        result
    }

    fn matrix_subtract(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
        let vec_result = m1.iter().zip(m2.iter()).map(|(&i1, &i2)| (i1 - i2)).collect();
        let result: DMatrix<f32> = DMatrix::from_vec(m1.nrows(), m1.ncols(), vec_result);
        result
    }

    fn matrix_mul(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
        m1 * m2
    }

    fn matrix_mul2(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
        let matrix1 = m1.clone();
        let matrix2 = m2.clone();
        let mut matrix3 = DMatrix::<f32>::zeros(m1.shape().0, m2.shape().1);
        for (i, x) in matrix1.row_iter().enumerate() {
            for (j, y) in matrix2.column_iter().enumerate() {
                matrix3[(i, j)] = (x * y).sum();
            }
        }
        matrix3
    }

    fn matrix_division(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
        m1.component_div(m2)
    }

    #[pyfn(m)]
    fn py_matrix_sum<'py>(_py: Python<'py>, m: &'py PyArrayDyn<f32>) -> f32 {
        let new_matrix = OMatrix::<f32, Dyn, Dyn>::from_vec(m.dims()[0], m.dims()[1], m.to_vec().unwrap()).transpose();
        new_matrix.sum()
    }

    #[pyfn(m)]
    fn py_matrix_reciprocal<'py>(py: Python<'py>, m: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
        let mut new_matrix = OMatrix::<f32, Dyn, Dyn>::from_vec(m.dims()[0], m.dims()[1], m.to_vec().unwrap()).transpose();
        new_matrix = matrix_reciprocal(&new_matrix);
        let result_matrix = new_matrix.data.as_vec().to_owned();
        let result = result_matrix.to_pyarray(py).reshape((m.dims()[0], m.dims()[1])).unwrap().to_dyn();
        result
    }

    #[pyfn(m)]
    fn py_matrix_subtract<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
        let new_matrix1 = OMatrix::<f32, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap()).transpose();
        let new_matrix2 = OMatrix::<f32, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap()).transpose();
        let new_matrix = new_matrix1 - new_matrix2;
        let result_matrix = new_matrix.data.as_vec().to_owned();
        let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
        result
    }

    #[pyfn(m)]
    fn py_matrix_multiply<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
        let new_matrix1 = OMatrix::<f32, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap()).transpose();
        let new_matrix2 = OMatrix::<f32, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap()).transpose();
        let new_matrix = new_matrix1.component_mul(&new_matrix2);
        let result_matrix = new_matrix.data.as_vec().to_owned();
        let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
        result
    }

    #[pyfn(m)]
    fn py_matrix_mul<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
        let matrix1 = OMatrix::<f32, Dyn, Dyn>::from_row_iterator(m1.dims()[0], m1.dims()[1], m1.to_owned_array().into_iter());
        let matrix2 = OMatrix::<f32, Dyn, Dyn>::from_row_iterator(m2.dims()[0], m2.dims()[1], m2.to_owned_array().into_iter());
        let new_matrix = matrix_mul(&matrix1, &matrix2).transpose();
        let result_matrix = new_matrix.data.as_vec().to_owned();
        let result = result_matrix.to_pyarray(py).reshape((m1.dims()[0], m2.dims()[1])).unwrap().to_dyn();
        result
    }

    #[pyfn(m)]
    fn py_matrix_division<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
        let matrix1 = OMatrix::<f32, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap()).transpose();
        let matrix2 = OMatrix::<f32, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap()).transpose();
        let new_matrix = matrix_division(&matrix1, &matrix2);
        let result_matrix = new_matrix.data.as_vec().to_owned();
        let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
        result
    }

    #[pyfn(m)]
    fn py_calculate_q<'py>(py: Python<'py>,
                           v: &'py PyArrayDyn<f32>, u: &'py PyArrayDyn<f32>,
                           w: &'py PyArrayDyn<f32>, h: &'py PyArrayDyn<f32>) -> f32 {
        let matrix_v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap());
        let matrix_u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap());
        let matrix_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap()).transpose();
        let matrix_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap()).transpose();
        calculate_q(&matrix_v, &matrix_u, &matrix_w, &matrix_h)
    }

    #[pyfn(m)]
    fn py_calculate_q_robust<'py>(py: Python<'py>,
                           v: &'py PyArrayDyn<f32>, u: &'py PyArrayDyn<f32>,
                           w: &'py PyArrayDyn<f32>, h: &'py PyArrayDyn<f32>, robust_alpha: f32) -> (f32, &'py PyArrayDyn<f32>) {
        let matrix_v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap());
        let matrix_u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap());
        let matrix_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap()).transpose();
        let matrix_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap()).transpose();
        let results = calculate_q_robust(&matrix_v, &matrix_u, &matrix_w, &matrix_h, robust_alpha);
        let q_robust = results.0;
        let results_uncertainty = results.1.transpose().data.as_vec().to_owned();
        let updated_uncertainty = results_uncertainty.to_pyarray(py).reshape(u.dims()).unwrap().to_dyn();
        (q_robust, updated_uncertainty)
    }
    Ok(())
}