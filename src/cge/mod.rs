//! An implementation of the CGE neural network encoding. The Network struct has methods for
//! evaluating a neural network, resetting its state, and saving to and loading from files.

mod utils;
pub mod gene;
pub mod network;

pub use self::network::Network;
