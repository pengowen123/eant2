// A trait with only a fitness function implemented by the user

use modules::network;

pub trait FitnessFunction {
    fn get_fitness(mut network: network::Network) -> f64;
}
