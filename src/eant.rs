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
// Add bias: Set bias field of an input to a tuple with a value and a weight (stored as a const)
//
// Add subnetwork: Add neuron gene and connect it to approx. 50% of inputs (use forward jumper
// genes)

use cge::node::*;

use cge::network::Network;
use cmaes::*;

pub fn eant_loop<T>(trait_dummy: T, threads: u8) -> Vec<f64>
	where T: FitnessFunction
{
    // Allow user to pass a neural network as an argument to generate the initial population
    // Ensure user input is valid
    // Multithreading does nothing at the moment, because each thread holds the mutex lock
    // for most of its lifetime, preventing other threads from doing anything; this problem
    // will go away once it is modified to be general
    // However, multithread the EANT2 code in the same fashion and set threads to 1 on the CMA-ES
    // calls
    // Also, use an options struct for both CMA-ES and EANT2, using method chaining, but also allow
    // default values ot make it easy to use
    let mut network = Network::new();
    network.genome = vec![Node::Neuron(Neuron {
        current_value: 0.0,
        weight: 1.0,
        input_count: 2,
        id_number: 0,
    }),

    Node::Neuron(Neuron {
        current_value: 0.0,
        weight: 1.0,
        input_count: 2,
        id_number: 1,
    }),

    Node::Neuron(Neuron {
        current_value: 0.0,
        weight: 1.0,
        input_count: 2,
        id_number: 3,
    }),
    
    Node::Input(Input {
        current_value: 0.0,
        weight: 1.0,
        id_number: 0,
    }),

    Node::Input(Input {
        current_value: 0.0,
        weight: 1.0,
        id_number: 1,
    }),

    Node::Input(Input {
        current_value: 0.0,
        weight: 1.0,
        id_number: 1,
    }),

    Node::Neuron(Neuron {
        current_value: 0.0,
        weight: 1.0,
        input_count: 3,
        id_number: 2,
    }),

    Node::Input(Input {
        current_value: 0.0,
        weight: 1.0,
        id_number: 0,
    }),

    Node::Input(Input {
        current_value: 0.0,
        weight: 1.0,
        id_number: 1,
    }),

    Node::JumperRecurrent(JumperRecurrent {
        weight: 1.0,
        id_number: 0
    })
];
   cmaes_loop(trait_dummy,
               network.clone(),
               threads,
               CMAESEndConditions::Stabilized(0.0000000000000000000000000000000000000000000000000000000000001, 10))
}
