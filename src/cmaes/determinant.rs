// nalgebra does not implement determinant for DMat

extern crate nalgebra as na;

use self::na::DMat;

use eant::functions::{permutations, prod_vec, sum_vec};

pub fn to_nested(mat: &DMat<f64>) -> Vec<Vec<f64>> {
	let rows = mat.nrows();
	let cols = mat.ncols();
	
	let mut new = Vec::new();
	
	for y in 0..rows {
		let mut vec = Vec::new();
		
		for x in 0..cols {
			vec.push(mat[(x, y)]);
		}
		
		new.push(vec);
	}
	
	new
}

pub fn determinant(mat: &DMat<f64>) ->  f64 {
	let mat = to_nested(mat);
	let size = mat.len();
	let perms = permutations(size);
	
	let mut vec = Vec::new();
	let mut sign = 1.0;
	
	for perm in perms {
		let mut mul = Vec::new();
		
		for i in 0..size {
			mul.push(mat[i][perm[i]]);
		}
		
		vec.push(sign * prod_vec(&mul));
		
		sign *= -1.0;
	}
	
	sum_vec(&vec)
}