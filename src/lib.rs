//! An implementation of the EANT2 algorithm used for training neural networks. The algorithm
//! requires very little information about the problem, and creates minimal, high fitness networks.
//!
//! While this algorithm is extremely general and can learn almost any task, it can take a very
//! long time to run. Because of this, it is sometimes necessary to adjust the options to make it
//! run faster. It is important to design a good task, otherwise the algorithm can take days to run.
//! Here are some tips for creating a good one:
//!
//! 1. Lower the amount of inputs and outputs to each neural network. Design the fitness function
//! to either do some of the work, or just simplify the problem.
//!
//! 2. Run the algorithm on good hardware to take advantage of the multithreaded support. The
//! algorithm is a one time thing; create a neural network with it and distribute and run it on
//! less powerful computers. On a related note, it is okay if it takes some time to run because it
//! will only have to run once.
//!
//! 3. Adjust individual options for the algorithm through `EANT2Options`. The documentation
//! provides a lot of information on how each option effects the algorithm.
//!
//! Note that it is important to get everything right the first time, otherwise the results might
//! be below expectation, which can mean running the algorithm again.
//!
//! # Examples
//!
//! Complete this section when the project is finished
// I couldn't care less about whether this implementation is inefficent. 99% of time spent is on
// CMA-ES, so it doesn't even matter.

extern crate cmaes;
extern crate cge;
extern crate rand;

mod utils;
mod cmaes_utils;
mod mutation_utils;
mod mutation;
mod generation;
mod compare;
mod cge_utils;
mod select;
pub mod options;
pub mod fitness;
pub mod eant;

pub use self::fitness::NNFitnessFunction;
pub use self::options::EANT2Options;
pub use self::cmaes::options::CMAESEndConditions;
pub use self::eant::eant_loop;
pub use self::cge::Network;
