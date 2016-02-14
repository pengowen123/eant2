// For testing parts of the crate

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
    use eant2::cge;
    use cge::gene::Gene;
    use cge::gene::GeneExtras::*;
    use cge::network::Network;

    // Create network from paper and make sure everything works fine
    // Also add Travis
    let network = Network {
        size: 11,
        genome: vec![
            Gene {
        ]
    };
}
