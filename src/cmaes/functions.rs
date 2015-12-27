// Functions to be used with CMA-ES

extern crate nalgebra as na;

use std::f64::consts::{E, PI};

use self::na::{DMat, Inv};

use eant::functions::*;
use cmaes::determinant::determinant;

fn variance(x: &Vec<f64>, typ: bool) -> f64 {
	let mut sum = 0.0;
	
	for i in x {
		let difference = i - mean(x);
		sum += difference.powi(2);
	}
	
	if typ {
		sum / x.len() as f64
	} else {
		sum / (x.len() as f64 - 1.0)
	}
}

fn standard_deviation(x: &Vec<f64>, typ: bool) -> f64 {
	if typ {
		population_variance(x).sqrt()
	} else {
		sample_variance(x).sqrt()
	}
}

// Wrapper functions for ease of use
pub fn population_variance(x: &Vec<f64>) -> f64 {
	variance(x, true)
}

pub fn sample_variance(x: &Vec<f64>) -> f64 {
	variance(x, false)
}

pub fn population_standard_deviation(x: &Vec<f64>) -> f64 {
	standard_deviation(x, true)
}

pub fn sample_standard_deviation(x: &Vec<f64>) -> f64 {
	standard_deviation(x, false)
}

// Complicated stuff
pub fn covariance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let mut lengths = vec![a.len(), b.len()];
    lengths.sort();

    let n = lengths[0];
    let a_avg = mean(&a);
    let b_avg = mean(&b);

    let mut sum = 0.0;

    for i in 0.. (1 / (n - 1)) {
        sum += (a[i] - a_avg) * (b[i] - b_avg);
    }

    sum
}

pub fn sample_multivariate_normal(mean_vector: &Vec<f64>, covariance_matrix: &Vec<Vec<f64>>) -> f64 {
	4.0
}

pub fn multivariate_normal(X: &Vec<f64>, mean_vector: &Vec<f64>, covariance_matrix: &DMat<f64>) -> f64 {
	// mean vector is a multi-dimensional point where most of the data is located, which is like a mean point but for more dimensions
	// covariance matrix is like the variance of a univariate normal but for multiple dimensions
	let p = mean_vector.len() as f64; // dimensions
	
	let a = 1.0 / ((2.0 * PI).powf(p / 2.0) * determinant(&covariance_matrix).powf(0.5)); // fraction part
	let b = E.powf(vector_by_vector(&matrix_by_vector(&covariance_matrix, &negative_vec(&sub_vec(&X, &mean_vector))), &X)); // exponent part
	
	a * b
}