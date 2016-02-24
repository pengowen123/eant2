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
    let options = EANT2Options::new(2, 2)
        .population_size(15)
        .offspring_count(2)
        .fitness_threshold(-1.0)
        .max_generations(1000)
        .cmaes_runs(1);

    let mut solution = eant_loop(&Foo, options).0;

    println!("{}", Foo.get_fitness(&mut solution));
}
