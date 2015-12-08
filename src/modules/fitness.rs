use modules::network;

pub trait FitnessFunction {
    fn get_fitness(network: network::Network) -> f64;
}
