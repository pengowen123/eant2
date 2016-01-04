// CMA-ES is used to optimize the connection weights of a generation of neural networks

// if possible make this concurrent, although it might be too complicated

// mean vector is the current best solution i think

extern crate nalgebra as na;

use std::f64::consts::{E, PI};
use std::thread;

use self::na::{DMat, Cov, Inv, Mean};

use eant::network::Network;
use eant::fitness::FitnessFunction;
use cmaes::functions::*;
use cmaes::network::NetworkCMAES;

pub fn cmaes_loop<T>(_: T, generation: &Vec<Network>, sample_size: i32, threads: i32)
    where T: FitnessFunction + Clone {

    let mut generation = NetworkCMAES::convert(generation);
    let mut covariance_matrix = vec![vec![0.0]];
    let mut mean_vector = vec![0.0];
    let mut step_size = 0.0;
    let mut end = false;

    while !end {
        for i in 0..sample_size {
            let i = i as usize;
            generation[i].thing = sample_multivariate_normal(&mean_vector, &covariance_matrix);
            generation[i].fitness = T::get_fitness(&mut generation[i].network);
        }
		
        generation.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // update state variables here
    }
}