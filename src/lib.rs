//! An implementation of the EANT2 algorithm used for training neural networks. The algorithm
//! requires very little information about the problem, and creates minimal, high fitness networks.
//!
//! # Examples
//!
//! Complete this section when the project is finished

extern crate cmaes;
extern crate cge;
extern crate rand;

mod utils;
mod compare;
mod mutationops;
mod select;
pub mod options;
pub mod fitness;
pub mod eant;

pub use self::fitness::NNFitnessFunction;
pub use self::options::EANT2Options;
pub use self::eant::eant_loop;
pub use self::cge::Network;
