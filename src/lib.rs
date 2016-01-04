// eant_rust is a library made for easy use of machine learning

pub use self::cge::network::Network;
pub use self::eant::fitness::FitnessFunction;
pub use self::eant::eant::eant_loop;

pub mod cge;
pub mod cmaes;
pub mod eant;
