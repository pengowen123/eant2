// CMA-ES is used to optimize the connection weights of a generation of neural networks

// if possible make this concurrent, although it might be too complicated

// something about the covariance matrix being (i, j) where i is the parameters of the network and
// j is the fitness, it is also symmetric maybe
// this means that the matrix is simply a measure of the effect each connection weight has on the
// fitness
// it is possible to adjust the weights based off of the matrix, similar to the way you train a
// perceptron
// this similarity might have something to do with them both being stochastic, whatever that means

use std::f64::consts::{E, PI};
use std::thread;

use eant::network::Network;
use eant::fitness::FitnessFunction;
use cmaes::functions::*;
use cmaes::network::NetworkCMAES;

pub fn start<T>(_: T, generation: &Vec<Network>, sample_size: i32, threads: i32)
    where T: FitnessFunction + Clone {

    let mut generation = NetworkCMAES::convert(generation);
    let mut covariance_matrix = vec![0.0];
    let mut mean = 0.0;
    let mut step_size = 0.0;
    let mut end = false;

    while !end {
        for i in 0..sample_size {
            let i = i as usize;
            generation[i].thing = sample_multivariate_normal(mean, &covariance_matrix);
            generation[i].fitness = T::get_fitness(&mut generation[i].network);
        }
		
        generation.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // update state variables here
    }
}