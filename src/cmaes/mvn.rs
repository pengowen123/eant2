// Multivariate normal sampling function

extern crate rand;

use la::Matrix;
use self::rand::distributions::{Normal, IndependentSample};
use self::rand::thread_rng;

use cge::functions::*;

pub fn sample_mvn(step_size: f64,
                  mean_vector: &Vec<f64>,
                  eigenvectors: &Matrix<f64>,
                  eigenvalues: &Matrix<f64>) -> Vec<f64> {

    let mut values = eigenvalues.clone();
    let d = mean_vector.len();
    let mut rng = thread_rng();
    let dist = Normal::new(0.0, 1.0);

    let random_values = (0..d).map(|_| {
        dist.ind_sample(&mut rng)
    }).collect();

    for i in 0..d {
        let v = values[(i, i)].powf(0.5);
        values.set(i, i, v);
    }

    values = values * eigenvectors;
    add_vec(&mul_vec(&matrix_by_vector(&values, &random_values), step_size), &mean_vector)
}
