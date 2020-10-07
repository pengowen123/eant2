use std::sync::Arc;

use cge::Network;
use cge::gene::Gene;
use cmaes::FitnessFunction;
use rand::thread_rng;
use rand::distributions::{IndependentSample, Range};

use crate::NNFitnessFunction;

// Stores additional information about a neural network, useful for mutation operators and
// selection
#[derive(Clone)]
pub struct Individual<T: NNFitnessFunction + Clone> {
    pub network: Network,
    // Stores the age of the genes, for setting initial standard deviation of the parameters, to make older
    // genes have a more local search (older genes tend to become stable after being optimized multiple
    // times)
    pub ages: Vec<usize>,
    pub inputs: usize,
    pub outputs: usize,
    pub next_id: usize,
    pub fitness: f64,
    pub object: Arc<T>,
    pub duplicates: usize,
    pub similar: usize
}

impl<T: NNFitnessFunction + Clone> Individual<T> {
    // Convenience constructor
    pub fn new(inputs: usize, outputs: usize, network: Network, object: Arc<T>) -> Individual<T> {
        Individual {
            ages: vec![0; network.size + 1],
            network: network,
            inputs: inputs,
            outputs: outputs,
            next_id: outputs,
            fitness: 0.0,
            object: object,
            duplicates: 0,
            similar: 0
        }
    }
}

// Implements the CMA-ES fitness function for Individual to make the library easier to use
// Sets the parameters of the neural network, calls the EANT2 fitness function, and resets the
// internal state
impl<T: NNFitnessFunction + Clone> FitnessFunction for Individual<T> {
    fn get_fitness(&self, parameters: &[f64]) -> f64 {
        let mut network = Network {
            size: self.network.size,
            genome: self.network.genome.iter().enumerate().map(|(i, gene)| {
                Gene {
                    weight: parameters[i],
                    .. gene.clone()
                }
            }).collect(),
            function: self.network.function.clone(),
        };

        network.clear_state();

        let object = self.object.clone();

        object.get_fitness(&mut network)
    }
}

// Everything I do seems to make the compiler complain about the trait not being implemented for a
// reference to T, so I make it easier by doing what it wants
impl<'a, T: NNFitnessFunction> NNFitnessFunction for &'a T {
    fn get_fitness(&self, network: &mut Network) -> f64 {
        (*self).get_fitness(network)
    }
}

pub fn weighted_choice(weights: &[usize; 4]) -> usize {
    let total = weights.iter().fold(0, |acc, i| acc + i);
    let n = Range::new(0, total).ind_sample(&mut thread_rng());
    let mut sum = 0;

    for (i, x) in weights.iter().enumerate() {
        if n >= sum && n < sum + x {
            return i;
        }

        sum += *x;
    }

    panic!("invalid weights");
}
