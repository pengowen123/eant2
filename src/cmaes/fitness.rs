// A trait with only a fitness function implemented by the user

use cge::network;

pub trait FitnessFunction {
    fn get_fitness(network: &mut network::Network) -> f64;
}
