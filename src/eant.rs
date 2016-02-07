// This is the main algorithm, which mutates neural networks, then uses CMA-ES to optimize them
// High fitness networks are kept to be mutated, while low fitness networks are discarded
// The loop ends when a solution is found

use cge::node::Node;
use cge::node::{Neuron, Input};

use cge::network::Network;
use cmaes::cmaes::cmaes_loop;
use cmaes::fitness::FitnessFunction;
use cmaes::condition::CMAESEndConditions;

pub fn eant_loop<T>(trait_dummy: T, threads: u8) -> Vec<f64>
	where T: FitnessFunction
{
    // Allow user to pass a neural network as an argument to generate the initial population
    // Ensure user input is valid
    let mut network = Network::new();
    network.genome = vec![Node::Neuron(Neuron {
        current_value: 0.0,
        weight: 1.0,
        input_count: 1,
        id_number: 0,
    }),

    Node::Input(Input {
        current_value: 0.0,
        weight: 1.0,
        id_number: 0,
    })];

    cmaes_loop(trait_dummy,
               network.clone(),
               threads,
               CMAESEndConditions::FitnessThreshold(0.0001))
}
