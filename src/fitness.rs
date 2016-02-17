//! The fitness function.
use cge::Network;

/// The fitness function used by the EANT2 algorithm. A lower fitness represents a better
/// individual. Implement it for a type, and pass the type to the eant_loop function. Use
/// the self argument to access fields of a struct, to factor other things into the
/// fitness calculation.
pub trait NNFitnessFunction {
    fn get_fitness(&self, network: &mut Network) -> f64;
}
