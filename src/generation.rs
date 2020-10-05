use std::sync::Arc;

use cge::{Network, Activation};
use rand::thread_rng;
use rand::distributions::{IndependentSample, Range};

use crate::utils::Individual;
use crate::cge_utils::Mutation;
use crate::NNFitnessFunction;

// Creates a generation of random, minimal neural networks
pub fn initialize_generation<T>(population_size: usize,
                                offspring_count: usize,
                                inputs: usize,
                                outputs: usize,
                                activation: Activation,
                                object: Arc<T>) -> Vec<Individual<T>>
    where T: NNFitnessFunction + Clone
{

    let mut rng = thread_rng();

    let mut generation = Vec::new();

    for _ in 0..population_size * (offspring_count + 1) {
        let mut network = Network {
            size: 0,
            genome: Vec::new(),
            function: activation.clone(),
        };

        for i in (0..outputs).rev() {
            network.add_subnetwork(i, 0, inputs)
        }

        network.size = network.genome.len() - 1;

        generation.push(Individual::new(inputs, outputs, network, object.clone()));
    }

    // Make sure all inputs are connected
    for individual in &mut generation {
        let range = Range::new(0, outputs);

        for i in 0..inputs {
            if individual.get_input_copies(i) == 0 {
                let id = range.ind_sample(&mut rng);
                let index = individual.network.get_neuron_index(id).unwrap() + 1;

                individual.add_input(i, index);
            }
        }
    }

    generation
}
