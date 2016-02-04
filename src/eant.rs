// This is the main algorithm, which mutates neural networks, then uses CMA-ES to optimize them
// High fitness networks are kept to be mutated, while low fitness networks are discarded
// The loop ends when a solution is found

use cge::node::Node;

use cge::network::Network;
use cmaes::cmaes::cmaes_loop;
use cmaes::fitness::FitnessFunction;

pub fn eant_loop<T>(trait_dummy: T, threads: u8) -> Vec<f64>
	where T: FitnessFunction
{
    let mut network = Network::new();
    network.genome = vec![Node::new_neuron(), Node::new_input()];
    cmaes_loop(trait_dummy, &mut network, threads)
}
