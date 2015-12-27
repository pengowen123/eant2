// This project should be modular, concurrent, easy to use, customizable, documented, and have
// tests.
// When it is finished and useful, publish it

// change vectors to arrays after completion
// also add better error handling
// #[inline] on functions
// remember to delete these attributes too
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

extern crate eant_rust;
extern crate nalgebra as na;

use eant_rust::fitness::FitnessFunction;
use eant_rust::network;

use eant_rust::eant::functions::*;
use eant_rust::cmaes::functions::multivariate_normal;

use na::{DMat, Mean};

/* Example implementation

struct Foo;

impl FitnessFunction for Foo {
	fn get_fitness(mut network: network::Network) -> f64 {
		4.0
	}
}

*/

fn main() {
	let matrix = DMat::from_row_vec(2, 2, &[1.0, 0.0, 0.0, 1.0]);
	let vector = vec![0.0, 0.50];
	
	println!("{:?}", multivariate_normal(&vec![0.0, 0.0], &vector, &matrix));
}
