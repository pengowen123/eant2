// Basic functions, mostly for vectors

use la::Matrix;

pub fn magnitude(vec: &Vec<f64>) -> f64 {
    let mut magnitude = 0.0;

    for n in vec {
        magnitude += n.powi(2);
    }

    if magnitude == 0.0 {
        0.0
    } else {
        magnitude.sqrt()
    }
}

pub fn normalize(vec: &Vec<f64>) -> Vec<f64> {
    let magnitude = magnitude(vec);
    let mut new = Vec::new();

    for n in vec {
        new.push(n / magnitude);
    }

    new
}

pub fn normalize_eigenvectors(mat: &mut Matrix<f64>) {
    let mut columns = Vec::new();

    for i in 0..mat.cols() {
        let mut cells = Vec::new();

        for k in 0..mat.rows() {
            cells.push(mat[(i, k)]);
        }

        columns.push(cells);
    }

    for vec in &mut columns {
        *vec = normalize(vec);
    }

    let mut new = Vec::new();

    for vec in columns {
        new.extend(vec);
    }

    *mat = Matrix::new(mat.rows(), mat.cols(), new);
    
}

pub fn concat(vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut new = Vec::new();

    for v in vecs {
        new.extend_from_slice(v);
    }

    new
}

pub fn reverse<T: Clone>(vec: &[T]) -> Vec<T> {
    if vec.len() == 0 {
        return vec.to_vec();
    }

    let mut new = Vec::new();
    let mut index = vec.len() - 1usize;

    loop {
        new.push(vec[index].clone());
        if index == 0usize {
            break;
        }
        index -= 1usize;
    }

    new
}

pub fn mul_vec(vec: &Vec<f64>, val: f64) -> Vec<f64> {
    let mut new = Vec::new();
    
    for n in vec {
        new.push(n * val);
    }

    new
}

pub fn sum_vec(vec: &Vec<f64>) -> f64 {
    let mut total = 0.0;

    for i in vec {
        total += *i;
    }

    total
}

pub fn add_vec(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut new = Vec::new();

    for i in 0..a.len() {
        new.push(a[i] + b[i]);
    }

    new
}

pub fn sub_vec(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut new = Vec::new();

    for i in 0..a.len() {
        new.push(a[i] - b[i]);
    }

    new
}

pub fn prod_vec(vec: &Vec<f64>) -> f64 {
    let mut total = 1.0;

    for i in vec {
        total *= *i;
    }

    total
}

pub fn div_vec(vec: &Vec<f64>, val: f64) -> Vec<f64> {
    let mut new = Vec::new();

    for n in vec {
        new.push(n / val);
    }

    new
}

pub fn negative_vec(vec: &Vec<f64>) -> Vec<f64> {
    vec.iter().map(|x| -x).collect::<Vec<f64>>()
}

pub fn matrix_by_vector(mat: &Matrix<f64>, vec: &Vec<f64>) -> Vec<f64> {
    let mut result = Vec::new();
    let mut rows = Vec::new();
    let n = vec.len();
    let w = mat.rows();
    let h = mat.cols();

    for y in 0..h {
        let mut row = Vec::new();

        for x in 0..w {
            row.push(mat[(x, y)]);
        }

        rows.push(row);
    }

    for row in rows {
        let mut sum = 0.0;

        for i in 0..n {
            sum += vec[i] * row[i];
        }

        result.push(sum);
    }

    result
}

pub fn vector_by_vector(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

pub fn mul_vec_2(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut new = Vec::new();

    for i in 0..a.len() {
        new.push(a[i] * b[i]);
    }

    new
}

pub fn transpose(mat: &Matrix<f64>) -> Matrix<f64> {
    let mut new = Matrix::zero(mat.cols(), mat.rows());

    for y in 0..mat.rows() {
        for x in 0..mat.cols() {
            new.set(x, y, mat.get(y, x));
        }
    }

    new
}
