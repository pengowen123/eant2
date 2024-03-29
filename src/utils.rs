use cge::gene::{Gene, NeuronId};
use cmaes::{DVector, ObjectiveFunction};

use std::sync::Arc;

use crate::cge_utils::{Network, NetworkView};
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
    /// Always `Some` after at least one optimization has been performed
    pub fitness: Option<f64>,
    pub object: Arc<T>,
    pub duplicates: usize,
    pub similar: usize,
}

impl<T: FitnessFunction + Clone> Individual<T> {
    // Convenience constructor
    pub fn new(inputs: usize, outputs: usize, network: Network, object: Arc<T>) -> Individual<T> {
        Individual {
            ages: vec![0; network.len()],
            network,
            inputs,
            outputs,
            fitness: None,
            object,
            duplicates: 0,
            similar: 0,
        }
    }

    /// Evaluates the `Individual` on the given set of weight parameters.
    fn eval(&mut self, x: &DVector<f64>) -> f64 {
        self.network.set_weights(x.as_slice()).unwrap();
        let view = NetworkView::new(&mut self.network);
        self.object.fitness(view)
    }
}

// Implements the CMA-ES fitness function for Individual to make the library easier to use
// Sets the parameters of the neural network, calls the EANT2 fitness function, and resets the
// internal state
impl<T: FitnessFunction + Clone> ObjectiveFunction for Individual<T> {
    fn evaluate(&mut self, x: &cmaes::DVector<f64>) -> f64 {
        self.eval(x)
    }
}

impl<'a, T: FitnessFunction + Clone> ObjectiveFunction for &'a mut Individual<T> {
    fn evaluate(&mut self, x: &cmaes::DVector<f64>) -> f64 {
        self.eval(x)
    }
}

/// Returns an iterator over the direct children of the neuron with the given ID, or panics if it
/// does not exist.
pub fn get_direct_children(network: &Network, id: NeuronId) -> impl Iterator<Item = &Gene<f64>> {
    // TODO: Subnetworks could be detected and skipped here, though it might not be much faster on
    //       average
    let range = network[id].subgenome_range();
    range.filter_map(move |i| {
        if network.parent_of(i).unwrap() == Some(id) {
            Some(&network[i])
        } else {
            None
        }
    })
}
