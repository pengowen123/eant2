// Basic functions, mostly for vectors

use la::Matrix;

pub fn reverse<T: Clone>(vec: &Vec<T>) -> Vec<T> {
    if vec.len() == 0 {
        return vec.clone();
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
    let mat = Matrix::new(2, 2, vec![1.0, 0.3, 0.3, 1.0]);
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

pub fn negative_vec(vec: &Vec<f64>) -> Vec<f64> {
    vec.iter().map(|x| -x).collect::<Vec<f64>>()
}

pub fn matrix_by_vector(mat: &Matrix<f64>, vec: &Vec<f64>) -> Vec<f64> {
    let mut result = Vec::new();
    let n = vec.len();
    let w = mat.rows();

    for i in 0..w {
        let mut sum = 0.0;
        let index = if i == 0 { 0 } else { i * n - 1 };
        let row = &mat.get_data()[index..index + n];

        for s in 0..vec.len() {
            sum += vec[s] * row[s];
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
