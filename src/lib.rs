//! An implementation of the EANT2 algorithm used for training neural networks.
//! It is easy to use, and works well on complex tasks.
//!
//! # Examples
//!
//! Complete this section when it the project is finished

extern crate cmaes;
extern crate rand;

pub use self::cge::network::Network;
pub use self::cmaes::fitness::FitnessFunction;
pub use self::eant::eant_loop;

pub mod cge;
pub mod eant;
