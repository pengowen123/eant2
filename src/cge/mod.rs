//! An implementation of the CGE neural network encoding. The Network struct has methods for
//! evaluating a neural network, and resetting its state. CGE can encode recurrent neural networks,
//! so calling the evaluate method multiple times without new inputs can still produce an output.
//! In the future, it will be possible to read a neural network from a file for portability.

mod utils;
mod gene;
pub mod network;

pub use self::network::Network;
