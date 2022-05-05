use crate::cge_utils::Mutation;
use crate::cmaes_utils::optimize_network;
use crate::eant2::EANT2;
use crate::utils::Individual;
use crate::FitnessFunction;
use cge::Network;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::sync::Arc;

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

        // preallocate the exact amount of memory we know we will need (using an object pool in an ergonomic way is hard here because `T` is unknown).
        let mut individuals = Vec::with_capacity(individual_count);
        let mut rng = thread_rng();

        (0..individual_count).for_each(|_| {
            // initialize empty network with the specified activation function
            let mut network = Network {
                size: 0,
                genome: vec![], // TODO: reuse buffer, guess at capacity on fallback to new alloc.
                function: options.activation.clone(),
            };

            // add subnetworks
            for i in (0..options.outputs).rev() {
                network.add_subnetwork(i, 0, options.inputs);
            }
            network.size = network.genome.len() - 1;

            // push individual into the generation
            individuals.push(Individual::new(
                options.inputs,
                options.outputs,
                network,
                object.clone(),
            ));
        });

        // ensure all inputs are connected
        individuals.iter_mut().for_each(|individual| {
            for i in 0..options.inputs {
                if individual.get_input_copies(i) == 0 {
                    let id = rng.gen_range(0..options.outputs);
                    let index = 1 + individual.network.get_neuron_index(id).unwrap();
                    individual.add_input(i, index);
                }
            }
        });

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
