// Multivariate normal sampling function

use std::f64::consts::{E, PI};

use la::Matrix;

use cge::functions::*;

// Complicated stuff
fn mvn(x: &Vec<f64>, mean_vector: &Vec<f64>, covariance_matrix: &Matrix<f64>) -> f64 {
    let p = mean_vector.len() as f64; // dimensions

    let a = 1.0 / ((2.0 * PI).powf(p / 2.0) * &covariance_matrix.det().powf(0.5)); // fraction part
    let b = E.powf(vector_by_vector(&matrix_by_vector(&covariance_matrix.inverse().unwrap(),
                                                      &negative_vec(&sub_vec(&x, &mean_vector))),
                                    &x)); // exponent part

    a * b
}

pub fn sample_mvn(mean_vector: &Vec<f64>, covariance_matrix: &Matrix<f64>) -> f64 {
    mvn(&vec![0.0, 0.0], mean_vector, covariance_matrix)
}