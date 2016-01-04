// Multivariate normal sampling function

extern crate nalgebra as na;

use std::f64::consts::{E, PI};

use self::na::{DMat, Inv};

use cge::functions::*;
use cmaes::determinant::determinant;

// Complicated stuff
fn mvn(x: &Vec<f64>, mean_vector: &Vec<f64>, covariance_matrix: &DMat<f64>) -> f64 {
    // mean vector is a multi-dimensional point where most of the data is located, which is like a mean point but for more dimensions
    // covariance matrix is like the variance of a univariate normal but for multiple dimensions
    let p = mean_vector.len() as f64; // dimensions

    let a = 1.0 / ((2.0 * PI).powf(p / 2.0) * determinant(&covariance_matrix).powf(0.5)); // fraction part
    let b = E.powf(vector_by_vector(&matrix_by_vector(&covariance_matrix.inv().unwrap(),
                                                      &negative_vec(&sub_vec(&x, &mean_vector))),
                                    &x)); // exponent part

    a * b
}

pub fn sample_mvn(mean_vector: &Vec<f64>, covariance_matrix: &DMat<f64>) -> f64 {
    mvn(&vec![0.0], mean_vector, covariance_matrix)
}
