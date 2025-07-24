extern crate core;

use std::collections::vec_deque::VecDeque;
use pyo3::prelude::*;
use pyo3::{PyObject, wrap_pyfunction, IntoPyObjectExt, IntoPyObject};
use pyo3::types::{PyTuple, PyFloat};
use numpy::{PyReadonlyArrayDyn, ToPyArray, PyArrayDyn, PyArray, PyArray2, PyArrayMethods, IntoPyArray};
use ndarray::{Array2};
use nalgebra::*;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget};
use std::io::{self, Write};
use console::Term;
use std::error::Error;

// GPU support requires the `candle` crate
use candle_core::{Device, Tensor, Result as CandleResult};


trait MatrixBackend: Send + Sync {
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>>;
    fn element_wise_divide(&self, a: &mut Array2<f64>, b: &Array2<f64>) -> Result<(), Box<dyn Error>>;
    fn element_wise_multiply(&self, a: &mut Array2<f64>, b: &Array2<f64>) -> Result<(), Box<dyn Error>>;
    fn should_use_gpu(&self, rows: usize, cols: usize) -> bool;
}

struct CpuBackend;

impl MatrixBackend for CpuBackend {
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        if a.ncols() != b.nrows() {
            return Err("Matrix dimensions do not match for multiplication".into());
        }
        Ok(a.dot(b))
    }
    fn element_wise_divide(&self, a: &mut Array2<f64>, b: &Array2<f64>) -> Result<(), Box<dyn Error>> {
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise division".into());
        }
        *a /= b;
        Ok(())
    }
    fn element_wise_multiply(&self, a: &mut Array2<f64>, b: &Array2<f64>) -> Result<(), Box<dyn Error>> {
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise multiplication".into());
        }
        *a *= b;
        Ok(())
    }
    fn should_use_gpu(&self, _rows: usize, _cols: usize) -> bool {
        false
    }
}

struct GpuBackend {
    device: Device,
    gpu_threshold: usize,
}
impl GpuBackend {
    fn new() -> Result<Self, Box<dyn Error>> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("Selected device: {:?}", device);
        Ok(Self {
            device,
            gpu_threshold: 1000, // adjust as needed
        })
    }
    fn array_to_tensor(&self, array: &Array2<f64>) -> CandleResult<Tensor> {
        let shape = array.shape();
        let data: Vec<f64> = array.iter().cloned().collect();
        Tensor::from_vec(data, (shape[0], shape[1]), &self.device)
    }
    fn tensor_to_array(&self, tensor: &Tensor) -> CandleResult<Array2<f64>> {
        let shape = tensor.shape();
        let data: Vec<f64> = tensor.to_vec2()?.into_iter().flatten().collect();
        let flat_data: Vec<f64> = data.into_iter().collect();
        Ok(Array2::from_shape_vec((shape.dims2()?.0, shape.dims2()?.1), flat_data).map_err(|e| candle_core::Error::msg(e.to_string()))?)
    }
}

impl MatrixBackend for GpuBackend {
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        if ! self.should_use_gpu(a.nrows().max(a.ncols()), b.nrows().max(b.ncols())) {
            return Ok(a.dot(b));
        }
        if a.ncols() != b.nrows() {
            return Err("Matrix dimensions do not match for multiplication".into());
        }
        let a_tensor = self.array_to_tensor(a)?;
        let b_tensor = self.array_to_tensor(b)?;
        let result_tensor = a_tensor.matmul(&b_tensor)?;
        self.tensor_to_array(&result_tensor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
    fn element_wise_divide(&self, a: &mut Array2<f64>, b: &Array2<f64>) -> Result<(), Box<dyn Error>> {
        if ! self.should_use_gpu(a.nrows(), a.ncols()){
            *a /= b;
            return Ok(());
        }
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise division".into());
        }
        let a_tensor = self.array_to_tensor(a)?;
        let b_tensor = self.array_to_tensor(b)?;
        let result_tensor = a_tensor.div(&b_tensor)?;
        *a = self.tensor_to_array(&result_tensor)?;
        Ok(())
    }
    fn element_wise_multiply(&self, a: &mut Array2<f64>, b: &Array2<f64>) -> Result<(), Box<dyn Error>> {
        if ! self.should_use_gpu(a.nrows(), a.ncols()) {
            *a *= b;
            return Ok(());
        }
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise multiplication".into());
        }
        let a_tensor = self.array_to_tensor(a)?;
        let b_tensor = self.array_to_tensor(b)?;
        let result_tensor = a_tensor.mul(&b_tensor)?;
        *a = self.tensor_to_array(&result_tensor)?;
        Ok(())
    }
    fn should_use_gpu(&self, rows: usize, cols: usize) -> bool {
        self.device.is_cuda() && (rows * cols) > self.gpu_threshold
    }
}
fn create_backend(prefer_gpu: bool) -> Box<dyn MatrixBackend> {
    if prefer_gpu {
        match GpuBackend::new() {
            Ok(gpu_backend) => {
                return Box::new(gpu_backend);
            }
            Err(_) => {
                println!("Warning: prefer_gpu=True but GPU is not available. Falling back to CPU backend.");
            }
        }
    }
    Box::new(CpuBackend)
}


#[pyfunction]
fn get_selected_device(prefer_gpu: Option<bool>) -> String {
    let prefer_gpu = prefer_gpu.unwrap_or(false);
    if prefer_gpu {
        match GpuBackend::new() {
            Ok(gpu_backend) => {
                if gpu_backend.device.is_cuda() {
                    return "cuda".to_string();
                } else {
                    return "cpu".to_string();
                }
            }
            Err(_) => return "cpu".to_string(),
        }
    }
    "cpu".to_string()
}

//--------------------- NMF Functions ---------------------//
const EPSILON: f32 = 1e-12;


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


#[pyfunction]
fn clear_screen() -> PyResult<()> {
    print!("\x1B[2J\x1B[H");
    io::stdout().flush().unwrap_or(());
    Ok(())
}

// NMF - Least-Squares Algorithm (LS-NMF)
// Returns (W, H, Q, converged)
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
    // Convert Python arrays to Array2<f32>
    let v_arr = v.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let u_arr = u.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let we_arr = we.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let w_arr = w.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let h_arr = h.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();

    // Call the Rust function with the converted arrays
    match ls_nmf_gpu(
        v_arr, u_arr, we_arr, w_arr, h_arr,
        max_iter, converge_delta, converge_n, robust_alpha, model_i,
        static_h, delay_h, progress_callback, prefer_gpu, py
    ) {
        Ok((final_w, final_h, q, converged, converge_i, q_list_full)) => {
            // Convert Rust Vecs back to numpy arrays
            let final_w = final_w.into_pyarray(py);
            let final_h = final_h.into_pyarray(py);
            let q_list_full_py = q_list_full.to_pyarray(py);
            let q_py = PyFloat::new(py, q as f64);

            Ok(PyTuple::new(
                py,
                [
                    final_w.into_pyobject(py),
                    final_h.into_pyobject(py),
                    q_py,
                    converged,
                    converge_i,
                    q_list_full_py.into_pyobject(py),
                ],
            )?.into_py_any(py)?)
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
}

fn ls_nmf_gpu<'py>(
    v: Array2<f32>,
    u: Array2<f32>,
    we: Array2<f32>,
    w: Array2<f32>,
    h: Array2<f32>,
    max_iter: i32,
    converge_delta: f32,
    converge_n: i32,
    robust_alpha: f32,
    model_i: i8,
    static_h: Option<bool>,
    delay_h: Option<i32>,
    progress_callback: Option<PyObject>,
    prefer_gpu: Option<bool>,
    py: Python<'py>,
) -> Result<(Vec<f32>, Vec<f32>, f32, bool, i32, Vec<f32>), Box<dyn Error>> {

    let backend = create_backend(prefer_gpu.unwrap_or(false));

    // Convert to f64 for backend trait compatibility
    let mut v64 = v.mapv(|x| x as f64);
    let mut u64 = u.mapv(|x| x as f64);
    let mut we64 = we.mapv(|x| x as f64);
    let mut w64 = w.mapv(|x| x as f64);
    let mut h64 = h.mapv(|x| x as f64);

    let hold_h = static_h.unwrap_or(false);
    let delay_h = delay_h.unwrap_or(-1);

    let mut q = 0.0;
    let mut converged = false;
    let mut converge_i = 0;
    let mut q_list: VecDeque<f32> = VecDeque::new();
    let mut q_list_full: VecDeque<f32> = VecDeque::new();
    let datapoints = v.len() as f32;

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

    for i in 0..max_iter {
        let wev64 = &we64 * &v64;

        if !hold_h || (delay_h > 0 && i > delay_h) {
            let wh = backend.matmul(&w64, &h64)?;
            let h_num = backend.matmul(&w64.t().to_owned(), &wev64)?;
            let h_den = backend.matmul(&w64.t().to_owned(), &(&we64 * &wh))?;
            let mut h_delta = h_num.clone();
            backend.element_wise_divide(&mut h_delta, &h_den)?;
            h64 = &h64 * &h_delta;
        }

        let wh = backend.matmul(&w64, &h64)?;
        let w_num = backend.matmul(&wev64, &h64.t().to_owned())?;
        let w_den = backend.matmul(&(&we64 * &wh), &h64.t().to_owned())?;
        let mut w_delta = w_num.clone();
        backend.element_wise_divide(&mut w_delta, &w_den)?;
        w64 = &w64 * &w_delta;

        // Calculate Q (sum of squared residuals, as in original)
        let wh = w64.dot(&h64);
        let residuals = &v64 - &wh;
        let weighted_residuals = &residuals / &u64;
        let qtrue = weighted_residuals.mapv(|x| x * x).sum() as f32;
        q = qtrue;
        let mse = qtrue / datapoints;
        q_list.push_back(q);
        q_list_full.push_back(q);
        converge_i = i;

        term.move_cursor_to(0, location_i).unwrap();
        pb.set_position((i + 1) as u64);
        term.move_cursor_to(0, location_i).unwrap();
        pb.set_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, qtrue, mse));

        if (q_list.len() as i32) >= converge_n {
            if q_list.front().unwrap() - q_list.back().unwrap() < converge_delta {
                converged = true;
            }
            q_list.pop_front();
        }
        if let Some(ref cb) = progress_callback {
            let completed = converged || (i + 1) == max_iter;
            let _ = cb.call1(py, (model_i, i, max_iter, qtrue, 0.0, mse, completed));
        }
        if converged {
            break;
        }
    }
    term.move_cursor_to(0, location_i).unwrap();
    pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, q, q / datapoints));

    // Convert back to f32 and flatten for Python
    let final_w = w64.mapv(|x| x as f32).into_raw_vec_and_offset().0;
    let final_h = h64.mapv(|x| x as f32).into_raw_vec_and_offset().0;
    let q_list_full_vec = q_list_full.into_iter().collect();

    Ok((final_w, final_h, q, converged, converge_i, q_list_full_vec))
}

// NMF - Weight Semi-NMF algorithm
// Returns (W, H, Q, converged)
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
) -> PyResult<PyObject> {
    let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.as_array().to_owned().into_raw_vec_and_offset().0);
    let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.as_array().to_owned().into_raw_vec_and_offset().0);
    let we = OMatrix::<f32, Dyn, Dyn>::from_vec(we.dims()[0], we.dims()[1], we.as_array().to_owned().into_raw_vec_and_offset().0);

    let mut new_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.as_array().to_owned().into_raw_vec_and_offset().0).transpose();
    let mut new_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.as_array().to_owned().into_raw_vec_and_offset().0).transpose();
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
            }
            q_list.pop_front();
        }
        if let Some(ref cb) = progress_callback {
            let completed = converged || (i + 1) == max_iter;
            let _ = cb.call1(py, (model_i, i, max_iter, qtrue, qrobust, mse, completed));
        }
        if converged {
            break
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

    let q_list_full_py = q_list_full.into_iter().collect::<Vec<_>>().to_pyarray(py);
    Ok((
        final_w,
        final_h,
        q,
        converged,
        converge_i,
        q_list_full_py,
    ).into_py(py))

}

// NMF - Weight Semi-NMF algorithm
// Returns (W, H, Q, converged)
#[pyfunction]
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
    progress_callback: Option<PyObject>,
) -> PyResult<PyObject> {
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
            }
            q_list.pop_front();
        }
        if let Some(ref cb) = progress_callback {
            let completed = converged || (i + 1) == max_iter;
            let _ = cb.call1(py, (model_i, i, max_iter, qtrue, qrobust, mse, completed));
        }
        if converged {
            break
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

    let q_list_full_py = q_list_full.into_iter().collect::<Vec<_>>().to_pyarray(py);
    Ok((
        final_w,
        final_h,
        q,
        converged,
        converge_i,
        q_list_full_py,
    ).into_py(py))
}

/// NMF - Multiplicative-Update (Kullback-Leibler)
/// Returns (W, H, Q, converged)
// #[pyfunction]
// fn nmf_kl<'py>(
//     py: Python<'py>,
//     v: PyReadonlyArrayDyn<'py, f32>, u: PyReadonlyArrayDyn<'py, f32>,
//     w: PyReadonlyArrayDyn<'py, f32>, h: PyReadonlyArrayDyn<'py, f32>,
//     update_weight: f32,
//     max_i: i32, converge_delta: f32, converge_i: i32
// ) -> PyResult<PyObject> {
//
//     let v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.to_vec().unwrap());
//     let u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.to_vec().unwrap());
//     let mut new_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.to_vec().unwrap()).transpose();
//     let mut new_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.to_vec().unwrap()).transpose();
//
//     let mut q: f32 = calculate_q(&v, &u, &new_w, &new_h);
//     let mut converged: bool = false;
//
//     let uv = v.component_div(&u);
//     let ur = matrix_reciprocal(&u);
//
//     let mut best_q: f32 = q;
//     let mut best_h = new_h.clone();
//     let mut best_w = new_w.clone();
//     let mut q_list: VecDeque<f32> = VecDeque::new();
//
//     let mut wh: DMatrix<f32>;
//     let mut h1: DMatrix<f32>;
//     let mut h2: DMatrix<f32>;
//     let mut w1: DMatrix<f32>;
//     let mut w2: DMatrix<f32>;
//     let mut best_i = 0;
//
//     let mut update_weight = update_weight;
//
//     let mut h_delta: DMatrix<f32>;
//     let mut w_delta: DMatrix<f32>;
//
//     for i in 0..max_i {
//         wh = &new_w * &new_h;
//         h1 = &new_w.transpose() * uv.component_div(&wh);
//         h2 = matrix_reciprocal(&(new_w.transpose() * &ur));
//         h_delta = update_weight * h2.component_mul(&h1);
//         new_h = new_h.component_mul(&h_delta);
//
//         wh = &new_w * &new_h;
//         w1 = uv.component_div(&wh) * &new_h.transpose();
//         w2 = matrix_reciprocal(&(&ur * &new_h.transpose()));
//         w_delta = update_weight * w2.component_mul(&w1);
//         new_w = new_w.component_mul(&w_delta);
//
//         q = calculate_q(&v, &u, &new_w, &new_h);
//         best_i = i;
//         if q < best_q {
//             best_q = q;
//             best_w = new_w.clone();
//             best_h = new_h.clone();
//         }
//         q_list.push_back(q);
//         if (q_list.len() as i32) >= converge_i {
//             let q_sum: f32 = q_list.iter().sum();
//             let q_avg: f32 = q_sum / q_list.len() as f32;
//             if (q_avg - q).abs() < converge_delta {
//                 if update_weight < 0.01 {
//                     converged = true;
//                     break
//                 }
//                 else {
//                     new_w = best_w.clone();
//                     new_h = best_h.clone();
//                     update_weight = &update_weight - 0.1;
//                     q_list.clear();
//                 }
//             }
//             else if (q_list.front().unwrap() - q_list.back().unwrap()) > 1.0 {
//                 new_w = best_w.clone();
//                 new_h = best_h.clone();
//                 update_weight = &update_weight - 0.1;
//                 q_list.clear();
//             }
//             q_list.pop_front();
//         }
//     }
//
//     best_w = best_w.transpose();
//     best_h = best_h.transpose();
//     let w_matrix = best_w.data.as_vec().to_owned();
//     let h_matrix = best_h.data.as_vec().to_owned();
//     let final_w = w_matrix.to_pyarray(py).reshape(w.dims()).unwrap().to_dyn();
//     let final_h = h_matrix.to_pyarray(py).reshape(h.dims()).unwrap().to_dyn();
//
//     let results = PyTuple::new(py, &[final_w, final_h, q.into_py(py), converged.into_py(py), converge_i.into_py(py), q_list_full.into_py(py)]);
//     Ok(results.into_py(py))
// }

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

// fn matrix_multiply(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
//     let vec_result = m1.iter().zip(m2.iter()).map(|(&i1, &i2)| (i1 * i2)).collect();
//     let result: DMatrix<f32> = DMatrix::from_vec(m1.nrows(), m1.ncols(), vec_result);
//     result
// }
//
// fn matrix_subtract(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
//     let vec_result = m1.iter().zip(m2.iter()).map(|(&i1, &i2)| (i1 - i2)).collect();
//     let result: DMatrix<f32> = DMatrix::from_vec(m1.nrows(), m1.ncols(), vec_result);
//     result
// }
//
// fn matrix_mul(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
//     m1 * m2
// }
//
// fn matrix_mul2(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
//     let matrix1 = m1.clone();
//     let matrix2 = m2.clone();
//     let mut matrix3 = DMatrix::<f32>::zeros(m1.shape().0, m2.shape().1);
//     for (i, x) in matrix1.row_iter().enumerate() {
//         for (j, y) in matrix2.column_iter().enumerate() {
//             matrix3[(i, j)] = (x * y).sum();
//         }
//     }
//     matrix3
// }
//
// fn matrix_division(m1: &DMatrix<f32>, m2: &DMatrix<f32>) -> DMatrix<f32> {
//     m1.component_div(m2)
// }

// #[pyfunction]
// fn py_matrix_sum<'py>(_py: Python<'py>, m: &'py PyArrayDyn<f32>) -> f32 {
//     let new_matrix = OMatrix::<f32, Dyn, Dyn>::from_vec(m.dims()[0], m.dims()[1], m.to_vec().unwrap()).transpose();
//     new_matrix.sum()
// }
//
// #[pyfunction]
// fn py_matrix_reciprocal<'py>(py: Python<'py>, m: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
//     let mut new_matrix = OMatrix::<f32, Dyn, Dyn>::from_vec(m.dims()[0], m.dims()[1], m.to_vec().unwrap()).transpose();
//     new_matrix = matrix_reciprocal(&new_matrix);
//     let result_matrix = new_matrix.data.as_vec().to_owned();
//     let result = result_matrix.to_pyarray(py).reshape((m.dims()[0], m.dims()[1])).unwrap().to_dyn();
//     result
// }
//
// #[pyfunction]
// fn py_matrix_subtract<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
//     let new_matrix1 = OMatrix::<f32, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap()).transpose();
//     let new_matrix2 = OMatrix::<f32, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap()).transpose();
//     let new_matrix = new_matrix1 - new_matrix2;
//     let result_matrix = new_matrix.data.as_vec().to_owned();
//     let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
//     result
// }

// #[pyfunction]
// fn py_matrix_multiply<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
//     let new_matrix1 = OMatrix::<f32, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap()).transpose();
//     let new_matrix2 = OMatrix::<f32, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap()).transpose();
//     let new_matrix = new_matrix1.component_mul(&new_matrix2);
//     let result_matrix = new_matrix.data.as_vec().to_owned();
//     let result = result_matrix.to_pyarray(py).reshape(m1.dims()).unwrap().to_dyn();
//     result
// }
//
// #[pyfunction]
// fn py_matrix_mul<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
//     let matrix1 = OMatrix::<f32, Dyn, Dyn>::from_row_iterator(m1.dims()[0], m1.dims()[1], m1.to_owned_array().into_iter());
//     let matrix2 = OMatrix::<f32, Dyn, Dyn>::from_row_iterator(m2.dims()[0], m2.dims()[1], m2.to_owned_array().into_iter());
//     let new_matrix = matrix_mul(&matrix1, &matrix2).transpose();
//     let result_matrix = new_matrix.data.as_vec().to_owned();
//     let result = result_matrix.to_pyarray(py).reshape((m1.dims()[0], m2.dims()[1])).unwrap().to_dyn();
//     result
// }
//
// #[pyfunction]
// fn py_matrix_division<'py>(py: Python<'py>, m1: &'py PyArrayDyn<f32>, m2: &'py PyArrayDyn<f32>) -> &'py PyArrayDyn<f32> {
//     let matrix1 = OMatrix::<f32, Dyn, Dyn>::from_vec(m1.dims()[0], m1.dims()[1], m1.to_vec().unwrap()).transpose();
//     let matrix2 = OMatrix::<f32, Dyn, Dyn>::from_vec(m2.dims()[0], m2.dims()[1], m2.to_vec().unwrap()).transpose();
//     let new_matrix = matrix_division(&matrix1, &matrix2);
//     let result_matrix = new_matrix.data.as_vec().to_owned();
//     let result = result_matrix.to_pyarray(py).reshape(m1.dims())?.to_dyn();
//     result
// }
//
// #[pyfunction]
// fn py_calculate_q<'py>(
//     _py: Python<'py>,
//     v: PyReadonlyArrayDyn<'py, f32>,
//     u: PyReadonlyArrayDyn<'py, f32>,
//     w: PyReadonlyArrayDyn<'py, f32>,
//     h: PyReadonlyArrayDyn<'py, f32>
// ) -> f32 {
//     let matrix_v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.as_array().to_owned().into_raw_vec_and_offset());
//     let matrix_u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.as_array().to_owned().into_raw_vec_and_offset());
//     let matrix_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.as_array().to_owned().into_raw_vec_and_offset()).transpose();
//     let matrix_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.as_array().to_owned().into_raw_vec_and_offset()).transpose();
//     calculate_q(&matrix_v, &matrix_u, &matrix_w, &matrix_h)
// }

// #[pyfunction]
// fn py_calculate_q_robust<'py>(
//     py: Python<'py>,
//     v: PyReadonlyArrayDyn<'py, f32>,
//     u: PyReadonlyArrayDyn<'py, f32>,
//     w: PyReadonlyArrayDyn<'py, f32>,
//     h: PyReadonlyArrayDyn<'py, f32>,
//     robust_alpha: f32
// ) -> (f32, PyObject) {
//     let matrix_v = OMatrix::<f32, Dyn, Dyn>::from_vec(v.dims()[0], v.dims()[1], v.as_array().to_owned().into_raw_vec_and_offset());
//     let matrix_u = OMatrix::<f32, Dyn, Dyn>::from_vec(u.dims()[0], u.dims()[1], u.as_array().to_owned().into_raw_vec_and_offset());
//     let matrix_w = OMatrix::<f32, Dyn, Dyn>::from_vec(w.dims()[1], w.dims()[0], w.as_array().to_owned().into_raw_vec_and_offset()).transpose();
//     let matrix_h = OMatrix::<f32, Dyn, Dyn>::from_vec(h.dims()[1], h.dims()[0], h.as_array().to_owned().into_raw_vec_and_offset()).transpose();
//
//     let results = calculate_q_robust(&matrix_v, &matrix_u, &matrix_w, &matrix_h, robust_alpha);
//     let q_robust = results.0;
//     let results_uncertainty = results.1.transpose().data.as_vec().to_owned();
//     let updated_uncertainty = results_uncertainty.to_pyarray(py);
//     (q_robust, updated_uncertainty)
// }

/// ESAT Rust module
#[pymodule]
fn esat_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(clear_screen, m)?)?;

    m.add_function(wrap_pyfunction!(ls_nmf, m)?)?;
    m.add_function(wrap_pyfunction!(ws_nmf, m)?)?;
    m.add_function(wrap_pyfunction!(ws_nmf_p, m)?)?;
    // m.add_function(wrap_pyfunction!(nmf_kl, m)?)?;

    // m.add_function(wrap_pyfunction!(py_matrix_reciprocal, m)?)?;
    // m.add_function(wrap_pyfunction!(py_matrix_sum, m)?)?;
    // m.add_function(wrap_pyfunction!(py_matrix_subtract, m)?)?;
    // m.add_function(wrap_pyfunction!(py_matrix_multiply, m)?)?;
    // m.add_function(wrap_pyfunction!(py_matrix_mul, m)?)?;
    // m.add_function(wrap_pyfunction!(py_matrix_division, m)?)?;
    //
    // m.add_function(wrap_pyfunction!(py_calculate_q, m)?)?;
    // m.add_function(wrap_pyfunction!(py_calculate_q_robust, m)?)?;

    m.add_function(wrap_pyfunction!(get_selected_device, m)?)?;

    Ok(())
}