extern crate core;

use std::time::Instant;
use std::error::Error;
use std::io::{self, Write};
use std::collections::vec_deque::VecDeque;
use console::Term;

use pyo3::prelude::*;
use pyo3::{PyObject, wrap_pyfunction};
use pyo3::types::{PyFloat, PyBool, PyInt, PyDict};
use numpy::{PyReadonlyArrayDyn, ToPyArray, IntoPyArray};
use ndarray::{Array2};
use nalgebra::*;
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget};


// GPU support requires the `candle` crate
use candle_core::{Device, Tensor, Result as CandleResult};


trait MatrixBackend: Send + Sync {
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>>;
    fn element_wise_divide(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>>;
    fn element_wise_multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>>;
    fn should_use_gpu(&self, rows: usize, cols: usize) -> bool;
    fn ls_nmf_update(
        &self,
        w: &Array2<f64>,
        h: &Array2<f64>,
        wev: &Array2<f64>,
        we: &Array2<f64>,
        hold_h: bool,
        delay_h: i32,
        iter: i32,
    ) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>>;
    fn ws_nmf_update(
        &self,
        w: &Array2<f64>,
        h: &Array2<f64>,
        wev: &Array2<f64>,
        we: &Array2<f64>,
        hold_h: bool,
        delay_h: i32,
        iter: i32,
    ) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>>;
}

struct CpuBackend;

impl MatrixBackend for CpuBackend {
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        if a.ncols() != b.nrows() {
            return Err("Matrix dimensions do not match for multiplication".into());
        }
        Ok(a.dot(b))
    }
    fn element_wise_divide(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise division".into());
        }
        let result = a / b;
        Ok(result)
    }
    fn element_wise_multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise multiplication".into());
        }
        let result = a * b;
        Ok(result)
    }
    fn should_use_gpu(&self, _rows: usize, _cols: usize) -> bool {
        false
    }
    fn ls_nmf_update(
        &self,
        w: &Array2<f64>,
        h: &Array2<f64>,
        wev: &Array2<f64>,
        we: &Array2<f64>,
        hold_h: bool,
        delay_h: i32,
        iter: i32,
    ) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
        let mut new_h = h.clone();
        let mut new_w = w.clone();

        let w_t = w.t().to_owned(); // Precompute transpose

        if !hold_h || (delay_h > 0 && iter > delay_h) {
            let wh = w.dot(h); // Compute once
            let h_num = w_t.dot(wev);
            let h_den = w_t.dot(&(we * &wh));
            new_h *= &(h_num / h_den); // Element-wise multiplication
        }
        let h_t = new_h.t().to_owned(); // Precompute transpose
        let wh = new_w.dot(&new_h); // Reuse updated `new_h`
        let w_num = wev.dot(&h_t);
        let w_den = (we * &wh).dot(&h_t);
        new_w *= &(w_num / w_den); // Element-wise multiplication

        Ok((new_w, new_h))
    }

    fn ws_nmf_update(
        &self,
        w: &Array2<f64>,
        h: &Array2<f64>,
        wev: &Array2<f64>,
        we: &Array2<f64>,
        hold_h: bool,
        delay_h: i32,
        iter: i32,
    ) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
        let mut new_w = w.clone();
        let mut new_h = h.clone();

        // Update W
        for (j, we_j) in we.rows().into_iter().enumerate() {
            let we_j_diag = Array2::from_diag(&we_j);
            let v_j = wev.row(j).to_owned();

            let w_n = new_h.dot(&v_j);
            let w_d = new_h.dot(&(we_j_diag.dot(&new_h.t())));
            let mut w_d_inv = ndarray_to_na(&w_d);
            if ! w_d_inv.try_inverse_mut() {
                w_d_inv = w_d_inv.pseudo_inverse(1e-10)?
            }
            let w_row = w_n.dot(&na_to_ndarray(&w_d_inv));
            new_w.row_mut(j).assign(&w_row);
        }

        // Update H
        if !hold_h || (delay_h > 0 && iter > delay_h) {
            let w_neg = (&new_w.mapv(|x| x.abs()) - &new_w) / 2.0;
            let w_pos = (&new_w.mapv(|x| x.abs()) + &new_w) / 2.0;

            for (j, we_j) in we.columns().into_iter().enumerate() {
                let we_j_diag = Array2::from_diag(&we_j);
                let v_j = wev.column(j).to_owned();

                let n1 = v_j.t().dot(&w_pos);
                let d1 = v_j.t().dot(&w_neg);

                let n2 = new_h.row(j).dot(&(w_neg.t().dot(&(we_j_diag.dot(&w_neg)))));
                let d2 = new_h.row(j).dot(&(w_pos.t().dot(&(we_j_diag.dot(&w_pos)))));

                let h_delta = (&n1 + &n2 + EPSILON) / (&d1 + &d2 + EPSILON);
                let current_column = new_h.column(j).to_owned();
                let updated_column = current_column * h_delta.mapv(|x| x.sqrt());
                new_h.column_mut(j).assign(&updated_column);
            }
        }

        Ok((new_w, new_h))
    }

}

struct GpuBackend {
    device: Device,
    gpu_threshold: usize,
}
impl GpuBackend {
    fn new() -> Result<Self, Box<dyn Error>> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("CUDA device available: {:?}", device);
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
    fn element_wise_divide(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        if ! self.should_use_gpu(a.nrows(), a.ncols()){
            let result = a / b;
            return Ok(result);
        }
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise division".into());
        }
        let a_tensor = self.array_to_tensor(a)?;
        let b_tensor = self.array_to_tensor(b)?;
        let result_tensor = a_tensor.div(&b_tensor)?;
        self.tensor_to_array(&result_tensor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)

    }
    fn element_wise_multiply(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        if ! self.should_use_gpu(a.nrows(), a.ncols()) {
            let result = a * b;
            return Ok(result);
        }
        if a.shape() != b.shape() {
            return Err("Shapes do not match for element-wise multiplication".into());
        }
        let a_tensor = self.array_to_tensor(a)?;
        let b_tensor = self.array_to_tensor(b)?;
        let result_tensor = a_tensor.mul(&b_tensor)?;
        self.tensor_to_array(&result_tensor).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
    fn should_use_gpu(&self, rows: usize, cols: usize) -> bool {
        self.device.is_cuda() && (rows * cols) > self.gpu_threshold
    }
    fn ls_nmf_update(
        &self,
        w: &Array2<f64>,
        h: &Array2<f64>,
        wev: &Array2<f64>,
        we: &Array2<f64>,
        hold_h: bool,
        delay_h: i32,
        iter: i32,
    ) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
        if !self.should_use_gpu(w.nrows().max(w.ncols()), h.nrows().max(h.ncols())) {
            // Fallback to CPU
            return CpuBackend.ls_nmf_update(w, h, wev, we, hold_h, delay_h, iter);
        }
        let w_tensor = self.array_to_tensor(w)?;
        let h_tensor = self.array_to_tensor(h)?;
        let wev_tensor = self.array_to_tensor(wev)?;
        let we_tensor = self.array_to_tensor(we)?;

        let mut new_h = h_tensor.clone();

        // Update H
        if !hold_h || (delay_h > 0 && iter > delay_h) {
            let wh = w_tensor.matmul(&h_tensor)?;
            let h_num = w_tensor.t().unwrap().matmul(&wev_tensor)?;
            let h_den = w_tensor.t().unwrap().matmul(&we_tensor.mul(&wh)?)?;
            let h_delta = h_num.div(&h_den)?;
            new_h = h_tensor.mul(&h_delta)?;
        }
        // Update W
        let wh = w_tensor.matmul(&new_h)?;
        let w_num = wev_tensor.matmul(&new_h.t().unwrap())?;
        let w_den = we_tensor.mul(&wh)?.matmul(&new_h.t().unwrap())?;
        let w_delta = w_num.div(&w_den)?;
        let new_w = w_tensor.mul(&w_delta)?;
        Ok((
            self.tensor_to_array(&new_w)?,
            self.tensor_to_array(&new_h)?
        ))
    }

    fn ws_nmf_update(
        &self,
        w: &Array2<f64>,
        h: &Array2<f64>,
        wev: &Array2<f64>,
        we: &Array2<f64>,
        hold_h: bool,
        delay_h: i32,
        iter: i32,
    ) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
        if !self.should_use_gpu(w.nrows().max(w.ncols()), h.nrows().max(h.ncols())) {
            // Fallback to CPU
            return CpuBackend.ws_nmf_update(w, h, wev, we, hold_h, delay_h, iter);
        }

        let w_tensor = self.array_to_tensor(w)?;
        let h_tensor = self.array_to_tensor(h)?;
        let wev_tensor = self.array_to_tensor(wev)?;
        let we_tensor = self.array_to_tensor(we)?;

        let mut new_w = w_tensor.clone();
        let mut new_h = h_tensor.clone();

        let EPS = Tensor::from_vec(vec![EPSILON], (1,), &self.device)?;

        // Update W
        for j in 0..we.shape()[0] {
            let we_j_diag: Tensor = create_diagonal(&we_tensor, j, &self.device).unwrap();
            let v_j: Tensor = wev_tensor.get(j)?;
            let w_n: Tensor = h_tensor.matmul(&v_j)?;
            let w_d: Tensor = h_tensor.matmul(&(we_j_diag.matmul(&h_tensor.t()?)?))?;
            let w_d_inv: Tensor = calculate_inverse(&w_d, &self.device)?;
            let w_row: Tensor = w_n.matmul(&w_d_inv)?;
            new_w.slice_set(&w_row, j, 0)?;
        }

        // Update H
        if !hold_h || (delay_h > 0 && iter > delay_h) {
            let w_neg = ((&new_w.abs()? - &new_w)? / 2.0)?;
            let w_pos =((&new_w.abs()? + &new_w)? / 2.0)?;

            for j in 0..we.shape()[1] {
                let we_j_diag = create_diagonal(&we_tensor, j, &self.device).unwrap();
                let v_j = wev_tensor.get(j)?;
                let n1 = v_j.t()?.matmul(&w_pos)?;
                let d1 = v_j.t()?.matmul(&w_neg)?;

                let n2 = new_h.get(j)?.matmul(&(w_neg.t()?.matmul(&(we_j_diag.matmul(&w_neg)?))?))?;
                let d2 = new_h.get(j)?.matmul(&(w_pos.t()?.matmul(&(we_j_diag.matmul(&w_pos)?))?))?;

                let h_delta = (&n1 + &n2)?.add(&EPS)? / (&d1 + &d2)?.add(&EPS)?;
                let new_h_col = new_h.get(j)?.reshape((new_h.shape().dims2()?.0, 1))?;
                new_h.slice_set(&new_h_col.mul(&h_delta?.sqrt()?)?, j, 1)?;
            }
        }

        Ok((
            self.tensor_to_array(&new_w)?,
            self.tensor_to_array(&new_h)?
        ))
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
const EPSILON: f64 = 1e-12;


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

    let mut q = Tensor::zeros((rows, cols), candle_core::DType::F32, device)?;
    let mut r = Tensor::zeros((cols, cols), candle_core::DType::F32, device)?;

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

// ESAT:NMF - Least-Squares Algorithm (LS-NMF)
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
    match ls_nmf_alg(
        v_arr, u_arr, we_arr, w_arr, h_arr,
        max_iter, converge_delta, converge_n, robust_alpha, model_i,
        static_h, delay_h, progress_callback, prefer_gpu, py
    ) {
        Ok((final_w, final_h, q, converged, converge_i, q_list_full)) => {
            // Convert Rust Vecs back to numpy arrays
            let final_w = final_w.into_pyarray(py);
            let final_h = final_h.into_pyarray(py);
            let q_list_full_py = q_list_full.to_pyarray(py);

            let py_results = PyDict::new(py);
            py_results.set_item("w", final_w)?;
            py_results.set_item("h", final_h)?;
            py_results.set_item("q", PyFloat::new(py, q as f64).into_any())?;
            py_results.set_item("converged", PyBool::new(py, converged))?;
            py_results.set_item("converge_i", PyInt::new(py, converge_i))?;
            py_results.set_item("q_list", q_list_full_py)?;

            Ok(py_results.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
}

fn ls_nmf_alg<'py>(
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
) -> Result<(Array2<f64>, Array2<f64>, f32, bool, i32, Vec<f32>), Box<dyn Error>> {

    let backend = create_backend(prefer_gpu.unwrap_or(false));

    // Convert to f64 for backend trait compatibility
    let v64 = v.mapv(|x| x as f64);
    let u64 = u.mapv(|x| x as f64);
    let we64 = we.mapv(|x| x as f64);
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

    let wev64 = backend.element_wise_multiply(&we64, &v64)?;
    for i in 0..max_iter {
        let (new_w, new_h) = backend.ls_nmf_update(&w64, &h64, &wev64, &we64, hold_h, delay_h, i)?;
        w64 = new_w;
        h64 = new_h;

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
            let now = Instant::now();
            if now.duration_since(last_callback) >= callback_interval || converged || (i + 1) == max_iter {
                let completed = converged || (i + 1) == max_iter;
                let _ = cb.call1(py, (model_i, i, max_iter, qtrue, 0.0, mse, completed));
                last_callback = now;
            }
        }
        if converged {
            break;
        }
    }
    term.move_cursor_to(0, location_i).unwrap();
    pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, q, q / datapoints));

    let final_w = w64.to_owned();
    let final_h = h64.to_owned();
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
    // Convert Python arrays to Array2<f32>
    let v_arr = v.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let u_arr = u.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let we_arr = we.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let w_arr = w.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let h_arr = h.as_array().to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();

    // Call the Rust function with the converted arrays
    match ws_nmf_alg(
        v_arr, u_arr, we_arr, w_arr, h_arr,
        max_iter, converge_delta, converge_n, robust_alpha, model_i,
        static_h, delay_h, progress_callback, prefer_gpu, py
    ) {
        Ok((final_w, final_h, q, converged, converge_i, q_list_full)) => {
            // Convert Rust Vecs back to numpy arrays
            let final_w = final_w.into_pyarray(py);
            let final_h = final_h.into_pyarray(py);
            let q_list_full_py = q_list_full.to_pyarray(py);

            let py_results = PyDict::new(py);
            py_results.set_item("w", final_w)?;
            py_results.set_item("h", final_h)?;
            py_results.set_item("q", PyFloat::new(py, q as f64).into_any())?;
            py_results.set_item("converged", PyBool::new(py, converged))?;
            py_results.set_item("converge_i", PyInt::new(py, converge_i))?;
            py_results.set_item("q_list", q_list_full_py)?;

            Ok(py_results.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
}

fn ws_nmf_alg<'py>(
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
) -> Result<(Array2<f64>, Array2<f64>, f32, bool, i32, Vec<f32>), Box<dyn Error>> {

    let backend = create_backend(prefer_gpu.unwrap_or(false));

    // Convert to f64 for backend trait compatibility
    let v64 = v.mapv(|x| x as f64);
    let u64 = u.mapv(|x| x as f64);
    let we64 = we.mapv(|x| x as f64);
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

    let wev64 = backend.element_wise_multiply(&we64, &v64)?;
    for i in 0..max_iter {
        let (new_w, new_h) = backend.ws_nmf_update(&w64, &h64, &wev64, &we64, hold_h, delay_h, i)?;
        w64 = new_w;
        h64 = new_h;

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
            let now = Instant::now();
            if now.duration_since(last_callback) >= callback_interval || converged || (i + 1) == max_iter {
                let completed = converged || (i + 1) == max_iter;
                let _ = cb.call1(py, (model_i, i, max_iter, qtrue, 0.0, mse, completed));
                last_callback = now;
            }
        }
        if converged {
            break;
        }
    }
    term.move_cursor_to(0, location_i).unwrap();
    pb.abandon_with_message(format!("Model: {}, Q(True): {:.4}, MSE: {:.4}", model_i, q, q / datapoints));

    let final_w = w64.to_owned();
    let final_h = h64.to_owned();
    let q_list_full_vec = q_list_full.into_iter().collect();

    Ok((final_w, final_h, q, converged, converge_i, q_list_full_vec))
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
    for (j, _col) in matrix1.column_iter().enumerate(){
        for (i, _row) in matrix1.row_iter().enumerate(){
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


/// ESAT Rust module
#[pymodule]
fn esat_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(clear_screen, m)?)?;

    m.add_function(wrap_pyfunction!(ls_nmf, m)?)?;
    m.add_function(wrap_pyfunction!(ws_nmf, m)?)?;

    m.add_function(wrap_pyfunction!(get_selected_device, m)?)?;

    Ok(())
}