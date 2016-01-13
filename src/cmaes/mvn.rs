// Multivariate normal sampling function

use la::{Matrix, EigenDecomposition};
use rand::random;

use cge::functions::*;

// Complicated stuff
pub fn sample_mvn(mean_vector: &Vec<f64>, covariance_matrix: &Matrix<f64>) -> Vec<f64> {
    let eigen = EigenDecomposition::new(covariance_matrix);
    
    let mut vectors = eigen.get_v().clone();
    let cols = vectors.cols();
    let rows = vectors.rows();
    {
        let mut data = vectors.get_mut_data();

        for i in 0..cols {
            let row = &mut data[i..i + rows];
            let mut len = 0.0;

            for n in row.iter() {
                len += n.powi(2);
            }

            len = len.powf(0.5);

            for n in row.iter_mut() {
                *n /= len;
            }
        }
    }

    let mut values = eigen.get_d();
    let d = mean_vector.len();
    
    let mut x = Vec::new();

    for _ in 0..d {
        let mut total = 0.0;

        for _ in 0..12 {
            total += random::<f64>();
        }

        x.push(total - 6.0);
    }

    for i in 0..d {
        let v = values[(i, i)].powf(0.5);
        values.set(i, i, v);
    }

    let mut q = Matrix::new(d, d, vec![0.0; d * d]);
    values.mmul(&vectors, &mut q);
    let y = add_vec(&matrix_by_vector(&q, &x), &mean_vector);

    y
}
