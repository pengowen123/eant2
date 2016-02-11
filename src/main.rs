// This project should be modular, concurrent, easy to use, customizable, documented, and have
// tests.
// When it is finished and useful, publish it

// also add better error handling
// #[inline] on functions
// remember to delete allow() attributes too

extern crate eant2;

use eant2::*;

// Example usage
struct Foo;

impl FitnessFunction for Foo {
    fn get_fitness(parameters: &[f64]) -> f64 {
        let solution = vec![10.0, 10.0];
        ((parameters[0] - solution[0]).powi(2) + (parameters[1] - solution[1]).powi(2)).sqrt()
    }
}

fn main() {
    let solved = eant_loop(Foo, 1);
    println!("solution: {:?}", solved);
    println!("fitness: {:?}", Foo::get_fitness(&solved));
}
