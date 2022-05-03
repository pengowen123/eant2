//! An implementation of the EANT2 algorithm used for training neural networks. The algorithm
//! requires very little information about the problem, and creates minimal, high fitness networks.
//!
//! While this algorithm is extremely general and can learn almost any task, it can take a very
//! long time to run. Because of this, it is sometimes necessary to adjust the options to make it
//! run faster. It is important to design a good task, otherwise the algorithm can take days to run.
//! Here are some tips for creating a good one:
//!
//! 1. Lower the amount of inputs and outputs to each neural network. Design the fitness function
//! to either do some of the work, or just simplify the problem. Another option is to train a
//! separate neural network to do processing of raw inputs, then feed its simplified output to the
//! EANT2 network.
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

// I couldn't care less about whether this implementation is efficent. 99% of time spent is on
// CMA-ES, so it doesn't even matter.

// FIXME: It appears CMA-ES is causing problems again. It spirals out of control, increasing
// fitness to an enormous value, eventually overflowing the stack (actually it doesn't overflow
// anymore for some reason). Please investigate.

// TODO: Test with the sigmoid transfer function, and a modified fitness function. The library may
// be broken because CMA-ES cannot correctly optimize a neural network to learn boolean logic.
// Fixing CMA-ES may solve this issue, but until then it would be good to know whether other types
// of neural networks can be trained. If they cannot, this library is broken, and should be fixed.

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
mod pools;
pub mod fitness;
pub mod eant2;
pub mod mutation_probabilities;

pub use self::cge::{Network, Activation};
pub use self::fitness::FitnessFunction;

mod tests;