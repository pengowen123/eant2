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
use eant_rust::cge::node;

extern crate la;
use la::Matrix;

// Example usage
struct Foo;

impl FitnessFunction for Foo {
    fn get_fitness(network: &mut Network) -> f64 {
        0.0
    }
}

fn main() {
    // Outputs zeroes if non-diagonal elements are 0.0
    let mut cov = Matrix::new(2, 2, vec![0.1, 0.0, 0.0, 0.1]);
    let mean = vec![0.0, 0.0];
    for _ in 0..100 {
       println!("{:?}", cmaes::mvn::sample_mvn(&mean, &cov));
    }
    let mut network = Network::new();
    let x = Foo::get_fitness(&mut network);
    eant_loop(Foo, 4);
}
