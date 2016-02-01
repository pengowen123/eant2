// Multivariate normal sampling function

use la::{Matrix, EigenDecomposition};
use rand::random;

use cge::functions::*;

pub fn sample_mvn(mean_vector: &Vec<f64>, covariance_matrix: &Matrix<f64>) -> Vec<f64> {
    let eigen = EigenDecomposition::new(covariance_matrix);
    
    let mut vectors = eigen.get_v().clone();
    let cols = vectors.cols();
    let rows = vectors.rows();

    normalize_eigenvectors(&mut vectors);

    let mut values = eigen.get_d();
    let d = mean_vector.len();
    let mut random_values = Vec::new();

    for _ in 0..d {
        let mut total = 0.0;

        for _ in 0..12 {
            total += random();
        }

        random_values.push(total - 6.0);
    }

    for i in 0..d {
        let v = values[(i, i)].powf(0.5);
        values.set(i, i, v);
    }

    values = values * vectors;
    add_vec(&matrix_by_vector(&values, &random_values), &mean_vector)
}
