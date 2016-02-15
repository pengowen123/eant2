use cge::Network;

/// A fitness function used by the EANT2 algorithm. Implement it for a type, and pass the type to
/// the eant_loop function.
pub trait NNFitnessFunction {
    fn get_fitness(network: &mut Network) -> f64;
}
