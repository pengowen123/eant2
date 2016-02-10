// This project should be modular, concurrent, easy to use, customizable, documented, and have
// tests.
// When it is finished and useful, publish it

// change vectors to arrays after completion
// also add better error handling
// #[inline] on functions
// remember to delete these attributes too
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

extern crate eant2;

use eant2::*;

// Example usage
struct Foo;

impl FitnessFunction for Foo {
    fn get_fitness(network: &mut Network) -> f64 {
        let data = vec![9.5, -4.5];
        let solution = -2.3;
        network.step(&data, false);
        let fitness = (network.step(&vec![0.2, 0.1], false)[0] - solution).abs();
        fitness
    }
}

fn main() {
    let solved = eant_loop(Foo, 1);
    println!("{:?}", solved);
}
