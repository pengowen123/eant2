// For testing parts of the crate
// TODO: Add Travis

extern crate eant2;

use eant2::*;

// Example usage
struct Foo;

impl NNFitnessFunction for Foo {
    fn get_fitness(network: &mut Network) -> f64 {
        let data = [vec![1.0], vec![2.0], vec![3.0]];

        let first = network.evaluate(&data[1])[0];
        let second = network.evaluate(&data[2])[0];
        let third = network.evaluate(&data[3])[0];

        let error = (first - 1.0).abs() + (second - 3.0).abs() + (third - 6.0).abs();

        error
    }
}

fn main() {

}
