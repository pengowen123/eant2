extern crate eant2;

use eant2::*;

#[derive(Clone)]
struct Foo;

impl NNFitnessFunction for Foo {
    fn get_fitness(&self, network: &mut Network) -> f64 {
        let answer = 3.14;

        let result = network.evaluate(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        (result.iter().fold(0.0, |acc, i| acc + i) - answer).abs()
    }
}

fn main() {
    let options = EANT2Options::new(1, 1)
        .population_size(5)
        .fitness_threshold(-1.0);

    eant_loop(&Foo, options);
}
