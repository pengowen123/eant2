//! Standard CMA-ES with multithreaded support.
//! // Implemented based on example code at:
//! en.wikipedia.org/wiki/CMA-ES
//! 
//! For now it is specifically designed for the EANT2 algorithm,
//! but a general use version will be released soon.

mod network;
pub mod fitness;
pub mod cmaes;
pub mod condition;

pub use self::cmaes::cmaes_loop;
pub use self::fitness::FitnessFunction;
pub use self::condition::CMAESEndConditions;
