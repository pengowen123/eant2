use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use cge::Network;
use cge::gene::Gene;
use cmaes::{ObjectiveFunction, DVector};
use rand::thread_rng;
use rand::distributions::{IndependentSample, Range};
use crate::FitnessFunction;

// Stores additional information about a neural network, useful for mutation operators and
// selection
#[derive(Clone)]
pub struct Individual<T: FitnessFunction + Clone> {
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

impl<T: FitnessFunction + Clone> Individual<T> {
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

    fn eval(&self, x: &DVector<f64>) -> f64 {
      // (copy the prospective parameters into the network)
      let mut network = Network {
        size: self.network.size,
        // TODO: there should an exist an API to evaluate the network
        // with a set of hypothetical parameters without committing to them
        // so we do not need to reallocate the genome for each hypothetical set of parameters.
        genome: {
          self.network.genome
            .iter()
            .zip(x.iter())
            .map(|(gene, &weight)| Gene { weight, ..gene.clone() })
            .collect()
        },
        function: self.network.function.clone(),
      };

      network.clear_state();

      self
        .object
        .clone()
        .get_fitness(&mut network)
    }
}

// Implements the CMA-ES fitness function for Individual to make the library easier to use
// Sets the parameters of the neural network, calls the EANT2 fitness function, and resets the
// internal state
impl<T: FitnessFunction + Clone> ObjectiveFunction for Individual<T> {
  fn evaluate(&mut self, x: &cmaes::DVector<f64>) -> f64 { self.eval(x) }
}

impl<'a, T: FitnessFunction + Clone> ObjectiveFunction for &'a Individual<T> {
  fn evaluate(&mut self, x: &cmaes::DVector<f64>) -> f64 { self.eval(x) }
}

impl<'a, T: FitnessFunction + Clone> ObjectiveFunction for &'a mut Individual<T> {
  fn evaluate(&mut self, x: &cmaes::DVector<f64>) -> f64 {  self.eval(x) }
}