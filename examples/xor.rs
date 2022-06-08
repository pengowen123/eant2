//! An example of using `eant2` to train a network to perform XOR logic.

extern crate cge;
extern crate eant2;

use cge::encoding::{Metadata, WithRecurrentState};
use eant2::eant2::EANT2;
use eant2::options::{EANT2Termination, Exploration};
use eant2::{Activation, FitnessFunction, NetworkView};


#[derive(Clone)]
struct Xor;

impl FitnessFunction for Xor {
    fn fitness(&self, mut network: NetworkView) -> f64 {
        let data = [
            ([0.0, 0.0], 0.0),
            ([1.0, 0.0], 1.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];

        let mut fitness = 0.0;

        for (input, expected_output) in data {
            let result = network.evaluate(&input).unwrap()[0];
            // Clear the state between each evaluation to prevent cheating using recurrent
            // connections
            network.clear_state();
            fitness += (result - expected_output).abs();
        }

        fitness
    }
}

fn main() {
    let eant = EANT2::builder()
        .inputs(2)
        .outputs(1)
        .activation(Activation::UnitStep)
        .print()
        .exploration(
            Exploration::builder()
                .terminate(
                    EANT2Termination::builder()
                        .fitness(0.01)
                        .build()
                )
                .build()
        )
        .build();

    let (network, _) = eant.run(&Xor);
    let metadata = Metadata::new(None);
    println!("{}", network.to_string(metadata, (), WithRecurrentState(false)).unwrap());
}
