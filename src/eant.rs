// This is the main algorithm, which mutates neural networks, then uses CMA-ES to optimize them
// High fitness networks are kept to be mutated, while low fitness networks are discarded
// The loop ends when a solution is found
//
// Eventually ship CMA-ES as its own crate, and add it as a dependency rather that having an
// internal copy
// 
// The goal is to automatically call network.set_parameters() and network.clear_genome()
// Ask a question on IRC or stackoverflow
//
// Mutation operators
//
// Add/Remove connection: Add/Delete gene and increment/decrement input count of the previous
// neuron
//
// Add bias: Set bias field of an input to a value (stored as a const)
//
// Add subnetwork: Add neuron gene and connect it to approx. 50% of inputs (use forward jumper
// genes)
//
// Move a minimal CGE to its own crate, add it as a dependency, and build onto it with traits
// including one with mutation operator methods

use cge::network::Network;
use cmaes::*;

pub fn eant_loop<T>(trait_dummy: T, threads: u8) -> Vec<f64>
	where T: FitnessFunction
{
    // Allow user to pass a neural network as an argument to generate the initial population
    // Ensure user input is valid

    let options = CMAESOptions::custom(2)
        .stable_generations(0.0000001, 10000);
    
    cmaes_loop(trait_dummy, options)
}
