// CMA-ES is used to optimize the connection weights of a generation of neural networks

// if possible make this concurrent, although it might be too complicated

// mean vector is the current best solution i think

extern crate la;

use std::thread;

use la::Matrix;

use cge::network::Network;
use cmaes::fitness::FitnessFunction;
use cmaes::network::NetworkCMAES;
use cmaes::mvn::sample_mvn;

pub fn cmaes_loop<T>(_: T, generation: &Vec<Network>, sample_size: u32, threads: u8)
    where T: FitnessFunction
{

    let mut generation = NetworkCMAES::convert(generation);
    let mut covariance_matrix = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let mut mean_vector = vec![0.0, 0.0];
    let mut step_size = 0.0;
    let mut end = false;

    while !end {
        for i in 0..sample_size {
            let i = i as usize;
            // generation[i].thing = sample_mvn(&mean_vector, &covariance_matrix);
            generation[i].fitness = T::get_fitness(&mut generation[i].network);
        }

        generation.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // update state variables here

        for i in 0..threads {
            thread::spawn(move || {
                println!("{}", i);
            });

            covariance_matrix = Matrix::new(3, 3, vec![0.0; 9]);
            mean_vector.push(0.0);
            step_size += 1.0;
            println!("foo {}", step_size);
            end = true;
        }
    }
}