// Functions related to matrices to be used with CMA-ES

// every covariance matrix is symmetric
// i, j position is the covariance between the i th and j th elements

use modules::functions::sum_vec;

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

pub fn sample_multivariate_normal() -> f64 {
    4.0
}
