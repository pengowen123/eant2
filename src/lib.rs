// eant_rust is a library made for easy use of machine learning

extern crate la;

pub use self::cge::network::Network;
pub use self::cmaes::fitness::FitnessFunction;
pub use self::eant::eant_loop;

pub mod cge;
pub mod cmaes;
pub mod eant;
