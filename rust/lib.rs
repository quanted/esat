extern crate core;

use std::time::Instant;
use std::error::Error;
use std::io::{self, Write};
use std::collections::vec_deque::VecDeque;
use console::Term;

use pyo3::prelude::*;
use pyo3::{PyObject, wrap_pyfunction};
use pyo3::types::{PyFloat, PyBool, PyInt, PyDict};
use numpy::{PyReadonlyArrayDyn, ToPyArray, IntoPyArray, PyArrayMethods};
use ndarray::{Array2};
use nalgebra::*;
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget};

// GPU support requires the `candle` crate
use candle_core::{Device, Tensor, Result as CandleResult};


// Convert nalgebra matrix (f32) to ndarray Array2<f64>
fn na_to_ndarray(mat: &OMatrix<f32, Dyn, Dyn>) -> Array2<f64> {
    let (rows, cols) = mat.shape();
    let data: Vec<f64> = mat.iter().map(|&x| x as f64).collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

// Convert ndarray Array2<f64> to nalgebra matrix (f32)
fn ndarray_to_na(arr: &Array2<f64>) -> OMatrix<f32, Dyn, Dyn> {
    let (rows, cols) = arr.dim();
    let data: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
    OMatrix::<f32, Dyn, Dyn>::from_vec(rows, cols, data)
}

fn create_diagonal(tensor: &Tensor, index: usize, device: &Device) -> CandleResult<Tensor> {
    let vector = tensor.get_on_dim(index, 0)?;
    let size = vector.shape().dims1()?;
    let mut diagonal_elements = vec![0.0; size * size];

    for i in 0..size {
        diagonal_elements[i * size + i] = vector.get(i)?.to_scalar()?;
    }

    Tensor::from_vec(diagonal_elements, (size, size), device)
}

fn calculate_inverse(tensor: &Tensor, device: &Device) -> CandleResult<Tensor> {
    let shape = tensor.shape().dims2()?;
    let (rows, cols) = (shape.0, shape.1);

    if rows != cols {
        return Err(candle_core::Error::msg("Matrix must be square for inverse calculation"));
    }

    // Calculate determinant
    let determinant = calculate_determinant(&tensor, device)?;
    if determinant.to_scalar::<f64>()? != 0.0 {
        // Calculate inverse using adjoint and determinant
        let adjoint = calculate_adjoint(&tensor, device)?;
        let inverse = adjoint.div(&Tensor::from_vec(vec![determinant.to_scalar::<f64>()?], (1,), device)?)?;
        return Ok(inverse);
    }

    // Pseudo-inverse using SVD decomposition
    let (u, s, vt) = calculate_svd(&tensor, device)?;
    let s_data = s.to_dtype(candle_core::DType::F64)?.to_vec1()?; // Convert tensor to a vector
    let s_inv_data: Vec<f64> = s_data
        .iter()
        .map(|&x: &f64| if x > 1e-12 { 1.0 / x } else { 0.0 })
        .collect();
    let s_inv = Tensor::from_vec(s_inv_data, s.shape().dims1()?, device)?; // Create GPU tensor
    let s_inv_diag = create_diagonal(&s_inv, 0, device)?;
    let pseudo_inverse = vt.matmul(&s_inv_diag)?.matmul(&u.t()?)?;
    Ok(pseudo_inverse)
}

fn calculate_adjoint(tensor: &Tensor, device: &Device) -> CandleResult<Tensor> {
    let shape = tensor.shape().dims2()?;
    let (rows, cols) = (shape.0, shape.1);

    if rows != cols {
        return Err(candle_core::Error::msg("Matrix must be square for adjoint calculation"));
    }

    let mut cofactor_matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let minor = extract_minor(&tensor, i, j, device)?;
            let cofactor = (-1.0f64).powi((i + j) as i32) * calculate_determinant(&minor, device)?.to_scalar::<f64>()?;
            cofactor_matrix[j * rows + i] = cofactor; // Transpose while calculating
        }
    }

    Tensor::from_vec(cofactor_matrix, (rows, cols), device)
}

fn calculate_determinant(tensor: &Tensor, device: &Device) -> CandleResult<Tensor> {
    let shape = tensor.shape().dims2()?;
    let (rows, cols) = (shape.0, shape.1);

    if rows != cols {
        return Err(candle_core::Error::msg("Matrix must be square for determinant calculation"));
    }

    if rows == 1 {
        let scalar = tensor.to_dtype(candle_core::DType::F64)?.get(0)?.to_scalar::<f64>()?;
        return Ok(Tensor::from_vec(vec![scalar], (1,), device)?);
    }

    let mut determinant = Tensor::from_vec(vec![0.0], (1,), device);
    for i in 0..cols {
        let minor = extract_minor(&tensor, 0, i, device)?;
        let cofactor = tensor.to_dtype(candle_core::DType::F64)?.get(0)?.get(i)?.to_scalar::<f64>()? * (-1.0f64).powi(i as i32);
        determinant = determinant?.add(&Tensor::from_vec(vec![cofactor], (1,), device)?.mul(&calculate_determinant(&minor, device)?)?);
    }

    Ok(determinant?)
}

fn calculate_svd(tensor: &Tensor, device: &Device) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let shape = tensor.shape().dims2()?;
    let (rows, cols) = (shape.0, shape.1);

    let mut u = Tensor::eye(rows, candle_core::DType::U32, device)?;
    let mut vt = Tensor::eye(cols, candle_core::DType::U32, device)?;
    let mut s = tensor.clone();

    for _ in 0..100 { // Iterative refinement
        let (q, r) = qr_decomposition(&s, device)?; // QR decomposition
        s = r.matmul(&q)?;
        u = u.matmul(&q)?;
        vt = q.matmul(&vt)?;
    }

    Ok((u, s, vt))
}

fn extract_minor(tensor: &Tensor, row: usize, col: usize, device: &Device) -> CandleResult<Tensor> {
    let shape = tensor.shape().dims2()?;
    let (rows, cols) = (shape.0, shape.1);

    if row >= rows || col >= cols {
        return Err(candle_core::Error::msg("Row or column index out of bounds"));
    }

    let mut minor_data = Vec::new();
    for i in 0..rows {
        if i == row {
            continue;
        }
        for j in 0..cols {
            if j == col {
                continue;
            }
            minor_data.push(tensor.to_dtype(candle_core::DType::F64)?.get(i)?.get(j)?.to_scalar::<f64>()?);
        }
    }

    Tensor::from_vec(minor_data, (rows - 1, cols - 1), device)
}

fn qr_decomposition(tensor: &Tensor, device: &Device) -> CandleResult<(Tensor, Tensor)> {
    let shape = tensor.shape().dims2()?;
    let (rows, cols) = (shape.0, shape.1);

    let q = Tensor::zeros((rows, cols), candle_core::DType::F32, device)?;
    let r = Tensor::zeros((cols, cols), candle_core::DType::F32, device)?;

    for i in 0..cols {
        let mut v = tensor.get(i)?;
        for j in 0..i {
            let q_col = q.get(j)?;
            let r_val = q_col.t()?.matmul(&v)?.to_scalar::<f64>()?;
            r.slice_set(&Tensor::from_vec(vec![r_val], (1,), device)?, j, i)?;
            v = v.sub(&q_col.mul(&Tensor::from_vec(vec![r_val], (1,), device)?)?)?;
        }
        let norm = v.sqr()?.sum(0)?.sqrt()?;
        r.slice_set(&norm, i, i)?;
        q.slice_set(&v.div(&norm)?, i, 1)?;
    }

    Ok((q, r))
}


#[pyfunction]
fn clear_screen() -> PyResult<()> {
    print!("\x1B[2J\x1B[H");
    io::stdout().flush().unwrap_or(());
    Ok(())
}

// ESAT:NMF - Least-Squares Algorithm (LS-NMF) - Python Function : Entrypoint
#[pyfunction]
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
    progress_callback: Option<PyObject>,
    prefer_gpu: Option<bool>,
) -> PyResult<PyObject> {

    // Determine device
    let prefer_gpu = prefer_gpu.unwrap_or(false);
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let use_gpu = prefer_gpu && device.is_cuda();

    // Convert Python arrays to ndarray Array2<f32>
    let v_arr = v.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let u_arr = u.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let we_arr = we.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let w_arr = w.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let h_arr = h.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();

    // Convert matrices to correct type for device
    let result = if use_gpu {
        // Convert to Candle Tensor using the selected device
        let v_t = Tensor::from_vec(
            v_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (v_arr.nrows(), v_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let u_t = Tensor::from_vec(
            u_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (u_arr.nrows(), u_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let we_t = Tensor::from_vec(
            we_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (we_arr.nrows(), we_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let w_t = Tensor::from_vec(
            w_arr.t().iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (w_arr.ncols(), w_arr.nrows()), // Transposed shape
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let h_t = Tensor::from_vec(
            h_arr.t().iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (h_arr.ncols(), h_arr.nrows()), // Transposed shape
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Call GPU update function (to be implemented)
        ls_nmf_update_gpu(
            v_t, u_t, we_t, w_t, h_t,
            max_iter, converge_delta, converge_n, robust_alpha, model_i,
            static_h, delay_h, progress_callback, py, device
        )
    } else {
        // Convert to nalgebra OMatrix
        let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap().to_owned());
        let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap().to_owned());
        let we = OMatrix::<f32, Dyn, Dyn>::from_vec(we.dims()[0], we.dims()[1], we.to_vec().unwrap().to_owned());

        let w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap().to_owned()).transpose();
        let h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap().to_owned()).transpose();

        // Call CPU update function (to be implemented)
        ls_nmf_update_cpu(
            v, u, we, w, h,
            max_iter, converge_delta, converge_n, robust_alpha, model_i,
            static_h, delay_h, progress_callback, py
        )
    };

    // Convert result to Python object
    match result {
        Ok((final_w, final_h, q, converged, converge_i, q_list_full)) => {
            let result_w = final_w.t().to_pyarray(py).reshape(w.dims())?;
            let result_h = final_h.t().to_pyarray(py).reshape(h.dims())?;

            let q_list_full_py = q_list_full.to_pyarray(py);

            let py_results = PyDict::new(py);
            py_results.set_item("w", result_w)?;
            py_results.set_item("h", result_h)?;
            py_results.set_item("q", PyFloat::new(py, q as f64).into_any())?;
            py_results.set_item("converged", PyBool::new(py, converged))?;
            py_results.set_item("converge_i", PyInt::new(py, converge_i))?;
            py_results.set_item("q_list", q_list_full_py)?;

            Ok(py_results.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
}

fn ls_nmf_update_cpu<'py>(
    v: OMatrix::<f32, Dyn, Dyn>,
    u: OMatrix::<f32, Dyn, Dyn>,
    we: OMatrix::<f32, Dyn, Dyn>,
    w: OMatrix::<f32, Dyn, Dyn>,
    h: OMatrix::<f32, Dyn, Dyn>,
    max_iter: i32,
    converge_delta: f32,
    converge_n: i32,
    robust_alpha: f32,
    model_i: i8,
    static_h: Option<bool>,
    delay_h: Option<i32>,
    progress_callback: Option<PyObject>,
    py: Python<'py>,
) -> Result<(Array2<f64>, Array2<f64>, f32, bool, i32, Vec<f32>), Box<dyn Error>> {

    let hold_h = static_h.unwrap_or(false);
    let delay_h = delay_h.unwrap_or(-1);

    let mut q = 0.0;
    let mut converged = false;
    let mut converge_i = 0;
    let mut q_list: VecDeque<f32> = VecDeque::new();
    let mut q_list_full: VecDeque<f32> = VecDeque::new();
    let datapoints = v.len() as f32;

    let location_i = (model_i as usize).try_into().unwrap();

    let mut last_update = Instant::now();
    let update_interval = std::time::Duration::from_secs_f32(1.0 / 30.0);

    let term = Term::buffered_stdout();
    term.move_cursor_to(0, location_i).unwrap();
    let draw_target = ProgressDrawTarget::term(term.clone(), 20);
    let pb = ProgressBar::with_draw_target(Some(max_iter as u64), draw_target);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {per_sec} iter/sec - {msg}")
            .unwrap()
            .progress_chars("-|-"),
    );

    let mut wh: DMatrix<f32>;
    let mut h_num: DMatrix<f32>;
    let mut h_den: DMatrix<f32>;
    let mut w_num: DMatrix<f32>;
    let mut w_den: DMatrix<f32>;

    let new_we = we.clone();
    let wev: DMatrix<f32> = we.component_mul(&v);

    let mut new_w = w.clone();
    let mut new_h = h.clone();

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

        let qtrue = calculate_q_cpu(&v, &u, &new_w, &new_h);
        q = qtrue;
        q_list.push_back(q);
        q_list_full.push_back(q);
        converge_i = i;

        if (q_list.len() as i32) >= converge_n {
            if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                converged = true;
            }
            q_list.pop_front();
        }

        let now = Instant::now();
        if now.duration_since(last_update) >= update_interval || converged || (i + 1) == max_iter {
            let qrobust = calculate_q_robust_cpu(&v, &u, &new_w, &new_h, robust_alpha);
            let mse = qtrue / datapoints;

            // Update progress bar and terminal output
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_position((i + 1) as u64);
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, qtrue, mse));

            if let Some(ref cb) = progress_callback {
                let completed = converged || (i + 1) == max_iter;
                let _ = cb.call1(py, (model_i, i, max_iter, qtrue, qrobust, mse, completed));
            }
            last_update = now;
        }
        if converged {
            break;
        }
    }
    term.move_cursor_to(0, location_i).unwrap();
    pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, q, q / datapoints));

    // Convert final matrices back to Array2<f64>
    let final_w = na_to_ndarray(&new_w);
    let final_h = na_to_ndarray(&new_h);
    let q_list_full_vec = q_list_full.into_iter().collect();

    Ok((final_w, final_h, q, converged, converge_i, q_list_full_vec))
}

fn ls_nmf_update_gpu<'py>(
    v: Tensor,
    u: Tensor,
    we: Tensor,
    mut w: Tensor,
    mut h: Tensor,
    max_iter: i32,
    converge_delta: f32,
    converge_n: i32,
    robust_alpha: f32,
    model_i: i8,
    static_h: Option<bool>,
    delay_h: Option<i32>,
    progress_callback: Option<PyObject>,
    py: Python<'py>,
    device: Device,
) -> Result<(Array2<f64>, Array2<f64>, f32, bool, i32, Vec<f32>), Box<dyn Error>> {
    let hold_h = static_h.unwrap_or(false);
    let delay_h = delay_h.unwrap_or(-1);

    let mut q = 0.0;
    let mut converged = false;
    let mut converge_i = 0;
    let mut q_list: VecDeque<f32> = VecDeque::new();
    let mut q_list_full: VecDeque<f32> = VecDeque::new();
    let datapoints = v.shape().elem_count() as f32;

    let location_i = (model_i as usize).try_into().unwrap();

    let mut last_update = Instant::now();
    let update_interval = std::time::Duration::from_secs_f32(1.0 / 60.0);

    let term = Term::buffered_stdout();
    term.move_cursor_to(0, location_i).unwrap();
    let draw_target = ProgressDrawTarget::term(term.clone(), 20);
    let pb = ProgressBar::with_draw_target(Some(max_iter as u64), draw_target);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {per_sec} iter/sec - {msg}")
            .unwrap()
            .progress_chars("-|-"),
    );

    // Precompute weighted V
    let wev = we.mul(&v)?;

    for i in 0..max_iter {
        // Update H
        if !hold_h || (delay_h > 0 && i > delay_h) {
            let wh = w.matmul(&h)?;
            let h_num = w.t()?.matmul(&wev)?;
            let h_den = w.t()?.matmul(&we.mul(&wh)?)?;
            let h_delta = h_num.div(&h_den)?;
            h = h.mul(&h_delta)?;
        }

        // Update W
        let wh = w.matmul(&h)?;
        let w_num = wev.matmul(&h.t()?)?;
        let w_den = we.mul(&wh)?.matmul(&h.t()?)?;
        let w_delta = w_num.div(&w_den)?;
        w = w.mul(&w_delta)?;

        let qtrue = calculate_q_gpu(&v, &u, &w, &h)?;
        q = qtrue;
        q_list.push_back(q);
        q_list_full.push_back(q);
        converge_i = i;

        if (q_list.len() as i32) >= converge_n {
            if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                converged = true;
            }
            q_list.pop_front();
        }

        let now = Instant::now();
        if now.duration_since(last_update) >= update_interval || converged || (i + 1) == max_iter {
            // For robust Q, use CPU fallback or implement a GPU version if needed
            let qrobust = calculate_q_robust_gpu(&v, &u, &w, &h, robust_alpha)?;
            let mse = qtrue / datapoints;

            term.move_cursor_to(0, location_i).unwrap();
            pb.set_position((i + 1) as u64);
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_message(format!("Model: {}, Q(True): {:?}, MSE: {:.4}", model_i, qtrue, mse));

            if let Some(ref cb) = progress_callback {
                let completed = converged || (i + 1) == max_iter;
                let _ = cb.call1(py, (model_i, i, max_iter, qtrue, qrobust, mse, completed));
            }
            last_update = now;
        }
        if converged {
            break;
        }
    }
    term.move_cursor_to(0, location_i).unwrap();
    pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, q, q / datapoints));

    // Convert final matrices back to Array2<f64>
    let final_w = {
        let w_f64 = w.to_dtype(candle_core::DType::F64)?;
        let w_vec2 = w_f64.to_vec2()?;
        let w_flat: Vec<f64> = w_vec2.into_iter().flatten().collect();
        Array2::from_shape_vec(w.shape().dims2()?, w_flat)?
    };
    let final_h = {
        let h_f64 = h.to_dtype(candle_core::DType::F64)?;
        let h_vec2 = h_f64.to_vec2()?;
        let h_flat: Vec<f64> = h_vec2.into_iter().flatten().collect();
        Array2::from_shape_vec(h.shape().dims2()?, h_flat)?
    };
    let q_list_full_vec = q_list_full.into_iter().collect();

    Ok((final_w, final_h, q, converged, converge_i, q_list_full_vec))
}


#[pyfunction]
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
    progress_callback: Option<PyObject>,
    prefer_gpu: Option<bool>,
) -> PyResult<PyObject> {
    // Determine device
    let prefer_gpu = prefer_gpu.unwrap_or(false);
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let use_gpu = prefer_gpu && device.is_cuda();

    // Convert Python arrays to ndarray Array2<f32>
    let v_arr = v.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let u_arr = u.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let we_arr = we.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let w_arr = w.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let h_arr = h.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();

    // Convert matrices to correct type for device
    let result = if use_gpu {
        // Convert to Candle Tensor using the selected device
        let v_t = Tensor::from_vec(
            v_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (v_arr.nrows(), v_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let u_t = Tensor::from_vec(
            u_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (u_arr.nrows(), u_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let we_t = Tensor::from_vec(
            we_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (we_arr.nrows(), we_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let w_t = Tensor::from_vec(
            w_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (w_arr.nrows(), w_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let h_t = Tensor::from_vec(
            h_arr.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            (h_arr.nrows(), h_arr.ncols()),
            &device
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Call GPU update function (to be implemented)
        ws_nmf_update_gpu(
            v_t, u_t, we_t, w_t, h_t,
            max_iter, converge_delta, converge_n, robust_alpha, model_i,
            static_h, delay_h, progress_callback, py, device
        )
    } else {
        // Convert to nalgebra OMatrix
        let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap());
        let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap());
        let we = OMatrix::<f32, Dyn, Dyn>::from_vec(we.dims()[0], we.dims()[1], we.to_vec().unwrap());

        let w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap()).transpose();
        let h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap()).transpose();

        // Call CPU update function (to be implemented)
        ws_nmf_update_cpu(
            v, u, we, w, h,
            max_iter, converge_delta, converge_n, robust_alpha, model_i,
            static_h, delay_h, progress_callback, py
        )
    };

    // Convert result to Python object
    match result {
        Ok((final_w, final_h, q, converged, converge_i, q_list_full)) => {
            let result_w = final_w.t().to_pyarray(py).reshape(w.dims())?;
            let result_h = final_h.t().to_pyarray(py).reshape(h.dims())?;
            let q_list_full_py = q_list_full.to_pyarray(py);

            let py_results = PyDict::new(py);
            py_results.set_item("w", result_w)?;
            py_results.set_item("h", result_h)?;
            py_results.set_item("q", PyFloat::new(py, q as f64).into_any())?;
            py_results.set_item("converged", PyBool::new(py, converged))?;
            py_results.set_item("converge_i", PyInt::new(py, converge_i))?;
            py_results.set_item("q_list", q_list_full_py)?;

            Ok(py_results.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
}

fn ws_nmf_update_cpu<'py>(
    v: OMatrix::<f32, Dyn, Dyn>,
    u: OMatrix::<f32, Dyn, Dyn>,
    we: OMatrix::<f32, Dyn, Dyn>,
    w: OMatrix::<f32, Dyn, Dyn>,
    h: OMatrix::<f32, Dyn, Dyn>,
    max_iter: i32,
    converge_delta: f32,
    converge_n: i32,
    robust_alpha: f32,
    model_i: i8,
    static_h: Option<bool>,
    delay_h: Option<i32>,
    progress_callback: Option<PyObject>,
    py: Python<'py>,
) -> Result<(Array2<f64>, Array2<f64>, f32, bool, i32, Vec<f32>), Box<dyn Error>> {

    let hold_h = static_h.unwrap_or(false);
    let delay_h = delay_h.unwrap_or(-1);

    let mut q = 0.0;
    let mut converged = false;
    let mut converge_i = 0;
    let mut q_list: VecDeque<f32> = VecDeque::new();
    let mut q_list_full: VecDeque<f32> = VecDeque::new();
    let datapoints = v.len() as f32;

    let location_i = (model_i as usize).try_into().unwrap();

    let mut last_callback = Instant::now();
    let callback_interval = std::time::Duration::from_secs_f32(1.0 / 30.0);

    let term = Term::buffered_stdout();
    term.move_cursor_to(0, location_i).unwrap();
    let draw_target = ProgressDrawTarget::term(term.clone(), 20);
    let pb = ProgressBar::with_draw_target(Some(max_iter as u64), draw_target);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {per_sec} iter/sec - {msg}")
            .unwrap()
            .progress_chars("-|-"),
    );

    let wev = we.component_mul(&v);
    let mut new_w = w.clone();
    let mut new_h = h.clone();

    // Precompute diagonal matrices for rows and columns of `we`
    let we_row_diags: Vec<DMatrix<f32>> = (0..we.nrows())
        .map(|j| DMatrix::from_diagonal(&DVector::from_row_slice(we.row(j).transpose().as_slice())))
        .collect();
    let we_col_diags: Vec<DMatrix<f32>> = (0..we.ncols())
        .map(|j| DMatrix::from_diagonal(&DVector::from_column_slice(we.column(j).as_slice())))
        .collect();

    let wev_mat = &wev;
    let vj_rows: Vec<DVector<f32>> = (0..wev_mat.nrows())
        .map(|j| DVector::from_row_slice(wev_mat.row(j).transpose().as_slice()))
        .collect();
    let vj_cols: Vec<DVector<f32>> = (0..wev_mat.ncols())
        .map(|j| DVector::from_row_slice(wev_mat.column(j).as_slice()))
        .collect();

    for i in 0..max_iter {
        // Update W
        let new_h_t = new_h.transpose();
        for j in 0..we.nrows() {
            let we_j_diag = &we_row_diags[j];
            let v_j = &vj_rows[j];
            let w_n = &new_h * v_j;
            let uh = we_j_diag * &new_h_t;
            let w_d = &new_h * uh;
            let w_d_inv = w_d.clone().try_inverse().unwrap_or_else(|| w_d.pseudo_inverse(1e-12).unwrap());
            let w_row = (w_n.transpose() * w_d_inv).row(0).transpose();
            new_w.set_row(j, &nalgebra::RowDVector::from_row_slice(w_row.as_slice()));
        }

        // Update H
        if !hold_h || (delay_h > 0 && i > delay_h) {
            let w_neg = (&new_w.abs() - &new_w) / 2.0;
            let w_pos = (&new_w.abs() + &new_w) / 2.0;

            for j in 0..we.ncols() {
                let we_j_diag = &we_col_diags[j];
                let v_j = &vj_cols[j];

                let n1 = v_j.transpose() * &w_pos;
                let d1 = v_j.transpose() * &w_neg;

                let n2a = w_neg.transpose() * we_j_diag;
                let n2b = &n2a * &w_neg;
                let d2a = w_pos.transpose() * we_j_diag;
                let d2b = &d2a * &w_pos;

                let h_j = new_h.column(j).transpose();
                let n2 = &h_j * &n2b;
                let d2 = &h_j * &d2b;
                let _n = (n1 + n2).add_scalar(1e-12);
                let _d = (d1 + d2).add_scalar(1e-12);
                let mut h_delta = _n.component_div(&_d);
                h_delta = h_delta.map(|x| x.sqrt());
                let _h = h_j.component_mul(&h_delta);
                let h_row = DVector::from_row_slice(_h.as_slice());
                new_h.set_column(j, &h_row);
            }
        }

        let qtrue = calculate_q_cpu(&v, &u, &new_w, &new_h);
        q = qtrue;
        q_list.push_back(q);
        q_list_full.push_back(q);
        converge_i = i;

        if (q_list.len() as i32) >= converge_n {
            if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                converged = true;
            }
            q_list.pop_front();
        }

        let now = Instant::now();
        if now.duration_since(last_callback) >= callback_interval || converged || (i + 1) == max_iter {
            let qrobust = calculate_q_robust_cpu(&v, &u, &new_w, &new_h, robust_alpha);
            let mse = qtrue / datapoints;

            term.move_cursor_to(0, location_i).unwrap();
            pb.set_position((i + 1) as u64);
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, qtrue, mse));

            if let Some(ref cb) = progress_callback {
                let completed = converged || (i + 1) == max_iter;
                let _ = cb.call1(py, (model_i, i, max_iter, qtrue, qrobust, mse, completed));
            }
            last_callback = now;
        }
        if converged {
            break;
        }
    }
    term.move_cursor_to(0, location_i).unwrap();
    pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, q, q / datapoints));

    // Convert final matrices back to Array2<f64> for output
    let final_w = na_to_ndarray(&new_w);
    let final_h = na_to_ndarray(&new_h);
    let q_list_full_vec = q_list_full.into_iter().collect();

    Ok((final_w, final_h, q, converged, converge_i, q_list_full_vec))
}

fn ws_nmf_update_gpu<'py>(
    v: Tensor,
    u: Tensor,
    we: Tensor,
    mut w: Tensor,
    mut h: Tensor,
    max_iter: i32,
    converge_delta: f32,
    converge_n: i32,
    robust_alpha: f32,
    model_i: i8,
    static_h: Option<bool>,
    delay_h: Option<i32>,
    progress_callback: Option<PyObject>,
    py: Python<'py>,
    device: Device
) -> Result<(Array2<f64>, Array2<f64>, f32, bool, i32, Vec<f32>), Box<dyn Error>> {
    let hold_h = static_h.unwrap_or(false);
    let delay_h = delay_h.unwrap_or(-1);

    let mut q = 0.0;
    let mut converged = false;
    let mut converge_i = 0;
    let mut q_list: VecDeque<f32> = VecDeque::new();
    let mut q_list_full: VecDeque<f32> = VecDeque::new();
    let datapoints = v.shape().elem_count() as f32;

    let location_i = (model_i as usize).try_into().unwrap();

    let mut last_callback = Instant::now();
    let callback_interval = std::time::Duration::from_secs_f32(1.0 / 30.0);

    let term = Term::buffered_stdout();
    term.move_cursor_to(0, location_i).unwrap();
    let draw_target = ProgressDrawTarget::term(term.clone(), 20);
    let pb = ProgressBar::with_draw_target(Some(max_iter as u64), draw_target);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {per_sec} iter/sec - {msg}")
            .unwrap()
            .progress_chars("-|-"),
    );

    let wev = we.mul(&v)?;

    for i in 0..max_iter {
        // Update W (vectorized, GPU-optimized)
        let wh = w.matmul(&h)?;
        let w_num = wev.matmul(&h.t()?)?;
        let w_den = we.mul(&wh)?.matmul(&h.t()?)?;
        let w_delta = w_num.div(&w_den)?;
        w = w.mul(&w_delta)?;

        // Update H (vectorized, GPU-optimized)
        if !hold_h || (delay_h > 0 && i > delay_h) {
            let w_abs = w.abs()?;
            let w_neg = (&w_abs - &w)?.mul(&Tensor::from_vec(vec![0.5], (1,), &device)?)?;
            let w_pos = (&w_abs + &w)?.mul(&Tensor::from_vec(vec![0.5], (1,), &device)?)?;

            let wh = w.matmul(&h)?;
            let h_num = w_pos.t()?.matmul(&wev)?;
            let h_den = w_neg.t()?.matmul(&wev)?;
            let n2a = w_neg.t()?.matmul(&we)?;
            let n2b = n2a.matmul(&w_neg)?;
            let d2a = w_pos.t()?.matmul(&we)?;
            let d2b = d2a.matmul(&w_pos)?;
            let h_j = h.t()?;
            let n2 = h_j.matmul(&n2b)?;
            let d2 = h_j.matmul(&d2b)?;
            let n = (h_num + n2)?.add(&Tensor::from_vec(vec![1e-12], (1,), &device)?)?;
            let d = (h_den + d2)?.add(&Tensor::from_vec(vec![1e-12], (1,), &device)?)?;
            let h_delta = n.div(&d)?.sqrt()?;
            h = h.mul(&h_delta.t()?)?;
        }

        let qtrue = calculate_q_gpu(&v, &u, &w, &h)?;
        q = qtrue;
        q_list.push_back(q);
        q_list_full.push_back(q);
        converge_i = i;

        if (q_list.len() as i32) >= converge_n {
            if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                converged = true;
            }
            q_list.pop_front();
        }

        let now = Instant::now();
        if now.duration_since(last_callback) >= callback_interval || converged || (i + 1) == max_iter {
            let qrobust = calculate_q_robust_gpu(&v, &u, &w, &h, robust_alpha)?;
            let mse = qtrue / datapoints;

            term.move_cursor_to(0, location_i).unwrap();
            pb.set_position((i + 1) as u64);
            term.move_cursor_to(0, location_i).unwrap();
            pb.set_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, qtrue, mse));

            if let Some(ref cb) = progress_callback {
                let completed = converged || (i + 1) == max_iter;
                let _ = cb.call1(py, (model_i, i, max_iter, qtrue, qrobust, mse, completed));
            }
            last_callback = now;
        }
        if converged {
            break;
        }
    }
    term.move_cursor_to(0, location_i).unwrap();
    pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, q, q / datapoints));

    // Convert final matrices back to Array2<f64>
    let final_w = {
        let w_f64 = w.to_dtype(candle_core::DType::F64)?;
        let w_vec2 = w_f64.to_vec2()?;
        let w_flat: Vec<f64> = w_vec2.into_iter().flatten().collect();
        Array2::from_shape_vec(w.shape().dims2()?, w_flat)?
    };
    let final_h = {
        let h_f64 = h.to_dtype(candle_core::DType::F64)?;
        let h_vec2 = h_f64.to_vec2()?;
        let h_flat: Vec<f64> = h_vec2.into_iter().flatten().collect();
        Array2::from_shape_vec(h.shape().dims2()?, h_flat)?
    };
    let q_list_full_vec = q_list_full.into_iter().collect();

    Ok((final_w, final_h, q, converged, converge_i, q_list_full_vec))
}


// Calculate Q (loss) for CPU
fn calculate_q_cpu(
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

// Calculate robust Q (loss) for CPU
fn calculate_q_robust_cpu(
    v: &OMatrix<f32, Dyn, Dyn>, u: &OMatrix<f32, Dyn, Dyn>,
    w: &OMatrix<f32, Dyn, Dyn>, h: &OMatrix<f32, Dyn, Dyn>,
    robust_alpha: f32
) -> f32 {
    let wh = w * h;
    let residuals = v - &wh;
    let scaled_residuals = residuals.component_div(u).abs();
    let robust_uncertainty = (&scaled_residuals / robust_alpha).map(|x| x.sqrt()).component_mul(&u);
    let robust_residuals = (residuals.component_div(&robust_uncertainty)).abs();
    let new_scaled_residuals = &scaled_residuals.clone();
    let merged_results = matrix_merge(&new_scaled_residuals, &robust_residuals, &u, &robust_uncertainty, robust_alpha);
    let merged_residuals = merged_results.0;
    // let updated_uncertainty = merged_results.1;
    let mr2 = merged_residuals.component_mul(&merged_residuals);
    let q_robust = mr2.sum();
    q_robust
}

// Calculate Q (loss) for GPU
fn calculate_q_gpu(
    v: &Tensor,
    u: &Tensor,
    w: &Tensor,
    h: &Tensor,
) -> CandleResult<f32> {
    let wh = w.matmul(h)?;
    let residuals = v.sub(&wh)?;
    let weighted_residuals = residuals.div(u)?;
    let wr2 = weighted_residuals.mul(&weighted_residuals)?;
    let q = wr2.sum_all()?.to_scalar::<f32>()?;
    Ok(q)
}

// Calculate robust Q (loss) for GPU
fn calculate_q_robust_gpu(
    v: &Tensor,
    u: &Tensor,
    w: &Tensor,
    h: &Tensor,
    robust_alpha: f32,
) -> CandleResult<f32> {
    let wh = w.matmul(h)?;
    let residuals = v.sub(&wh)?;
    let scaled_residuals = residuals.div(u)?.abs()?;
    let robust_uncertainty = scaled_residuals.div(&Tensor::from_vec(vec![robust_alpha as f64], (1,), v.device())?)?.sqrt()?.mul(u)?;
    let robust_residuals = residuals.div(&robust_uncertainty)?.abs()?;
    // Merge: use scaled_residuals where abs(scaled_residuals) < robust_alpha, else robust_residuals
    let mask = scaled_residuals.lt(&Tensor::from_vec(vec![robust_alpha as f64], (1,), v.device())?)?;
    let merged_residuals = mask.where_cond(&scaled_residuals, &robust_residuals)?;
    let mr2 = merged_residuals.mul(&merged_residuals)?;
    let q_robust = mr2.sum_all()?.to_scalar::<f32>()?;
    Ok(q_robust)
}


fn matrix_merge<'a>(
    matrix1: &'a OMatrix<f32, Dyn, Dyn>,
    matrix2: &'a OMatrix<f32, Dyn, Dyn>,
    matrix4: &'a OMatrix<f32, Dyn, Dyn>,
    matrix5: &'a OMatrix<f32, Dyn, Dyn>,
    alpha: f32
    ) -> (OMatrix<f32, Dyn, Dyn>, OMatrix<f32, Dyn, Dyn>) {
    let (rows, cols) = matrix1.shape();
    let mut matrix3 = OMatrix::<f32, Dyn, Dyn>::zeros(rows, cols);
    let mut matrix6 = OMatrix::<f32, Dyn, Dyn>::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            if matrix1[(i, j)] > alpha {
                matrix3[(i, j)] = matrix2[(i, j)];
                matrix6[(i, j)] = matrix5[(i, j)];
            } else {
                matrix3[(i, j)] = matrix1[(i, j)];
                matrix6[(i, j)] = matrix4[(i, j)];
            }
        }
    }
    (matrix3, matrix6)
}


/// ESAT Rust module
#[pymodule]
fn esat_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(clear_screen, m)?)?;

    m.add_function(wrap_pyfunction!(ls_nmf, m)?)?;
    m.add_function(wrap_pyfunction!(ws_nmf, m)?)?;

    Ok(())
}