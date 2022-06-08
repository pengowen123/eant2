use cge::gene::{Gene, Input, InputId, Neuron, NeuronId};
use cge::Activation;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use std::collections::HashSet;
use std::sync::Arc;

use crate::cge_utils::{Network, INITIAL_WEIGHT_VALUE};
use crate::cmaes_utils::optimize_network;
use crate::eant2::EANT2;
use crate::utils::Individual;
use crate::FitnessFunction;

pub struct Generation<T: FitnessFunction + Clone> {
    /// The individuals in the generation.
    // TODO: reuse buffer, guess at capacity on fallback to new alloc.
    // TODO: this will require something nice and clever to handle the fact that `T` is unknown.
    pub(crate) individuals: Vec<Individual<T>>,
}

impl<T: FitnessFunction + Clone> Generation<T> {
    /// Creates a generation of random, minimal neural networks.
    pub fn initialize(options: &EANT2, object: Arc<T>) -> Generation<T> {
        let individual_count = options.exploration.population * (1 + options.exploration.offspring);

        let individuals = (0..individual_count)
            .map(|_| {
                let network =
                    get_random_initial_network(options.inputs, options.outputs, options.activation);

                Individual::new(options.inputs, options.outputs, network, object.clone())
            })
            .collect();

        Generation { individuals }
    }

    /// Use `CMA-ES` to optimize all the individuals in the generation in parallel, exploiting their existing structure.
    pub fn update_generation(&mut self, options: &EANT2)
    where
        T: 'static + FitnessFunction + Clone + Send + Sync,
    {
        self.individuals
            .par_iter_mut()
            .for_each(|individual| optimize_network(individual, &options));
    }
}

/// Returns a random, minimal network with the specified number of inputs and outputs.
fn get_random_initial_network(inputs: usize, outputs: usize, activation: Activation) -> Network {
    let mut rng = thread_rng();

    // Generate the genome randomly
    let genome = (0..outputs)
        .map(|id| NeuronId::new(id))
        .flat_map(|neuron_id| {
            // Create a random subgenome for each network output
            let mut subgenome = Vec::new();

            let create_input = |id| Gene::from(Input::new(id, INITIAL_WEIGHT_VALUE));

            // Each subgenome is connected to ~50% of network inputs
            for input_id in (0..inputs).map(|id| InputId::new(id)) {
                if rng.gen() {
                    subgenome.push(create_input(input_id));
                }
            }

            // If no input genes were added, add a random input gene
            if subgenome.is_empty() {
                let input_id = InputId::new(rng.gen_range(0..inputs));
                subgenome.push(create_input(input_id));
            }

            // Add the subgenome's root neuron
            let subgenome_inputs = subgenome.len();
            let neuron = Neuron::new(neuron_id, subgenome_inputs, INITIAL_WEIGHT_VALUE).into();
            subgenome.insert(0, neuron);

            subgenome
        })
        .collect();

    let mut network = Network::new(genome, activation).unwrap();

    // Ensure that all network inputs are connected to a neuron
    let mut input_ids_not_connected = (0..inputs).collect::<HashSet<_>>();

    for g in network.genome() {
        if let Gene::Input(input) = g {
            input_ids_not_connected.remove(&input.id().as_usize());
        }
    }

    for input_id in input_ids_not_connected
        .into_iter()
        .map(|id| InputId::new(id))
    {
        // Add unconnected network inputs to a random output neuron
        let parent_id = NeuronId::new(rng.gen_range(0..outputs));
        network
            .add_non_neuron(parent_id, Input::new(input_id, INITIAL_WEIGHT_VALUE))
            .unwrap();
    }

    network
}
