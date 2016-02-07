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

extern crate eant_rust;

extern crate la;
use la::{Matrix, EigenDecomposition};

use eant_rust::*;
use eant_rust::cge::node::*;

// Example usage
struct Foo;

impl FitnessFunction for Foo {
    fn get_fitness(network: &mut Network) -> f64 {
        let coords: Vec<f64> = network.genome.iter().map(|n| {
            match *n {
                Node::Neuron(Neuron { ref weight, .. }) => *weight,
                Node::Input(Input { ref weight, .. }) => *weight,
                Node::JumperRecurrent(JumperRecurrent { ref weight, .. }) => *weight,
                Node::JumperForward(JumperForward { ref weight, .. }) => *weight
            }
        }).collect();

        let solution = 1.0;
        let result = (((0.5 * coords[0]) + (1.5 * coords[1])) - solution).abs();
        result
    }
}

fn main() {
    let solved = eant_loop(Foo, 1);
    println!("{:?}", solved);
}
