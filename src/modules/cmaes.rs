// CMA-ES is used to optimize the connection weights of a generation of neural networks

// if possible make this concurrent

// something about the covariance matrix being (i, j) where i is the parameters of the network and
// j is the fitness
// this means that the matrix is simply a measure of the effect each connection weight has on the
// fitness
// it is possible to adjust the weights based off of the matrix, similar to the way you train a
// perceptron
// this similarity might have something to do with them both being stochastic, whatever that means

use std::f64::consts::{E, PI};

use modules::network::Network;
use modules::fitness::FitnessFunction;
use modules::matrix::*;

struct Item {
    network: Network,
    fitness: i32,
    thing: f64
}

fn start<T>(generation: Vec<Item>,
            fitness_function: T,
            sample_size: i32,
            )
    where T: FitnessFunction + Clone {

    let mean = 0.0;
    let covariance_matrix = vec![0];

    let end = false;

    while !end {
        for i in 0..sample_size {

        }
    }
}
