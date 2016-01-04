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

use eant_rust::*;

struct Foo;

impl FitnessFunction for Foo {
	fn get_fitness(network: &mut Network) -> f64 { 0.0 }
}

fn main() {
	let mut network = Network::new();
	let x = Foo::get_fitness(&mut network);
	eant_loop(4);
}
