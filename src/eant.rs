// TODO: Adjust initial covariance matrix based on age of individual genes.
// Only adjust covariance matrix diagonals, because each represents the variance for a univariate
// normal distribution, and the covariance matrix is just multiple normal distributions.
//
// The goal is to automatically call network.set_parameters() and network.clear_genome()
// Ask a question on IRC or stackoverflow
//
// Mutation operators
// read paper for more info
//
// Add/Remove connection: Add/Delete gene and increment/decrement input count of the previous
// neuron
//
// Add bias: Add bias gene which stores a weight, and has a value stored in a const
//
// Add subnetwork: Add neuron gene and connect it to approx. 50% of inputs (use forward jumper
// genes)
//
// Add wrapper structs to allow a vector of only Inputs and Neurons, Forward and Recurrent Jumpers,
// etc. to easily index things that share a property for mutations. Either this or several vectors
// of indexes of genes available for certain operations.

use cge::Network;
use cmaes::*;
use rand::random;

use utils::{Individual, GeneAge};
use options::EANT2Options;
use mutationops::Mutation;
use fitness::NNFitnessFunction;

struct CMAESFitnessDummy;

impl FitnessFunction for CMAESFitnessDummy {
    fn get_fitness(parameters: &[f64]) -> f64 {
        0.0
    }
}

pub fn eant_loop<T>(trait_dummy: T, options: EANT2Options) -> Vec<f64>
	where T: NNFitnessFunction
{
    // Allow user to pass a neural network as an argument to generate the initial population
    // Ensure user input is valid
    
    // to make new generation, add new individuals to the population by mutating every individual
    // use the select function to rank and select which individuals go to the next generation
    // refer to the flow chart in the eant2 paper for more info
    // keep this file as abstract as possible
    // fix cmaes to allow for the todo feature

    let mut cmaes_options = CMAESOptions::custom(2);

    cmaes_options.end_conditions = vec![options.cmaes_end_condition.clone()];
    
    cmaes_loop(CMAESFitnessDummy, cmaes_options)
}
