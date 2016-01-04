// eant_rust is a library made for easy use of machine learning

pub use self::eant::network::Network;
pub use self::eant::fitness::FitnessFunction;
pub use self::eant::eant::eant_loop;

pub mod eant;
pub mod cmaes;
