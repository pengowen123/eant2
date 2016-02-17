use std::sync::Arc;

use cge::Network;
use cge::gene::Gene;
use cmaes::FitnessFunction;

use fitness::NNFitnessFunction;

#[derive(Clone)]
pub struct Individual<T: NNFitnessFunction + Clone> {
    pub network: Network,
    pub genes: Vec<GeneAge>,
    pub fitness: f64,
    pub object: Arc<T>,
    pub duplicates: usize,
    pub similar: usize
}

#[derive(Clone)]
pub struct GeneAge {
    pub gene: Gene,
    pub age: usize
}

impl<T: NNFitnessFunction + Clone> Individual<T> {
    pub fn new(network: Network, object: Arc<T>) -> Individual<T> {
        Individual {
            genes: network.genome.iter().map(|g| GeneAge { gene: g.clone(), age: 0 }).collect(),
            network: network,
            fitness: 0.0,
            object: object,
            duplicates: 0,
            similar: 0
        }
    }
}

impl<T: NNFitnessFunction + Clone> FitnessFunction for Individual<T> {
    fn get_fitness(&self, parameters: &[f64]) -> f64 {
        let mut network = Network {
            size: self.network.size,
            genome: self.network.genome.iter().enumerate().map(|(i, gene)| {
                Gene {
                    weight: parameters[i],
                    .. gene.clone()
                }
            }).collect()
        };

        network.clear_state();

        let object = self.object.clone();

        (*object).get_fitness((&mut network))
    }
}

impl<'a, T: NNFitnessFunction> NNFitnessFunction for &'a T {
    fn get_fitness(&self, network: &mut Network) -> f64 {
        (*self).get_fitness(network)
    }
}
