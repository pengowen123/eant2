// Basic functions, mostly for vectors

extern crate nalgebra as na;

use self::na::{DMat, RowSlice};

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

pub fn negative_vec(vec: &Vec<f64>) -> Vec<f64> {
	vec.iter().map(|x| -x).collect::<Vec<f64>>()
}

pub fn matrix_by_vector(mat: &DMat<f64>, vec: &Vec<f64>) -> Vec<f64> {
	let mut result = Vec::new();
	let n = vec.len();
	let w = mat.nrows();
	
	for i in 0..w {
		let mut sum = 0.0;
		let row = mat.row_slice(i, 0, n);
		
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

pub fn mean(x: &Vec<f64>) -> f64 {
	sum_vec(x) / x.len() as f64
}

pub fn permutations(n: usize) -> Vec<Vec<usize>> {
	let mut result = Vec::new();
	let mut start = (0..n).collect::<Vec<usize>>();
	
	result.push(start.clone());
	
	loop {
		let next = next_permutation(start.clone());
		
		if compare(&*start, &*next) {
			break;
		}
		
		result.push(next.clone());
		start = next;
	}
	
	result
}

fn next_permutation(mut sequence: Vec<usize>) -> Vec<usize> {
	let mut k = -1;
	let mut s = 0;
	
	for i in 0..sequence.len() {
		if sequence[i] < match sequence.get(i + 1) {
				Some(x) => *x,
				None => break
		} {
			k = i as i32;
		}
	}
	
	if k == -1 {
		return sequence;
	}
	
	for i in 0..sequence.len() {
		if sequence[i] > sequence[k as usize] {
			s = i as i32;
		}
	}
	
	let k = k as usize;
	let s = s as usize;
	
	sequence.swap(k, s);
	
	let mut slice = &sequence[k + 1..].to_vec();
	let slice = reverse(&slice);
	
	let mut si = 0;
	
	for i in k + 1..sequence.len() {
		sequence[i] = slice[si];
		si += 1;
	}
	
	sequence
}

fn compare(a: &[usize], b: &[usize]) -> bool {
	if a.len() != b.len() {
		return false;
	}
	
	for i in 0..a.len() {
		if a[i] != b[i] {
			return false;
		}
	}
	
	true
}