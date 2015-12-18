// A trait with only a fitness function implemented by the user

use eant::network;

pub trait FitnessFunction {
    fn get_fitness(mut network: &mut network::Network) -> f64;
}
