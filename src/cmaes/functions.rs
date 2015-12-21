// Functions to be used with CMA-ES

use std::f64::consts::{E, PI};

use eant::functions::sum_vec;

fn variance(x: &Vec<f64>, typ: bool) -> f64 {
	let mut sum = 0.0;
	
	for i in x {
		let difference = i - mean(x);
		sum += difference * difference;
	}
	
	// Population variance
	if typ {
		sum / x.len() as f64
	}
	// Sample variance
	else {
		sum / (x.len() as f64 - 1.0)
	}
}

fn standard_deviation(x: &Vec<f64>, typ: bool) -> f64 {
	// Population standard deviation
	if typ {
		population_variance(x).sqrt()
	}
	// Sample Standard deviation
	else {
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

pub fn mean(x: &Vec<f64>) -> f64 {
	sum_vec(x) / x.len() as f64
}

pub fn covariance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let mut lengths = vec![a.len(), b.len()];
    lengths.sort();

    let n = lengths[0];
    let a_avg = sum_vec(&a) / a.len() as f64;
    let b_avg = sum_vec(&b) / b.len() as f64;

    let mut sum = 0.0;

    for i in 0.. (1 / (n - 1)) {
        sum += (a[i] - a_avg) * (b[i] - b_avg);
    }

    sum
}

pub fn multivariate_normal(mean_vector: Vec<f64>, covariance_matrix: Vec<Vec<f64>>) -> f64 {
	0.0
}

pub fn sample_multivariate_normal(mean: f64, covariance_matrix: &Vec<f64>) -> f64 {
    4.0
}
