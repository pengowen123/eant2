// CMA-ES is used to optimize the connection weights of a generation of neural networks

// if possible make this concurrent, although it might be too complicated

extern crate la;

use std::thread;

use la::Matrix;

use cge::network::Network;
use cmaes::fitness::FitnessFunction;
use cmaes::network::NetworkParameters;
use cmaes::mvn::sample_mvn;

pub fn cmaes_loop<T>(_: T, network: &mut Network, sample_size: u32, threads: u8)
    where T: FitnessFunction
{
    let mut generation = vec![NetworkParameters::new(vec![1.0, 2.0]); sample_size as usize]; 
    let mut covariance_matrix = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let mut mean_vector = vec![0.0, 0.0];
    let mut step_size = 0.0;
    let mut end = false;

    while !end {
        for i in 0..sample_size as usize {
            generation[i] = NetworkParameters::new(sample_mvn(&mean_vector, &covariance_matrix));
            network.set_parameters(&generation[i].parameters);
            generation[i].fitness = T::get_fitness(network);
            network.clear_genome();
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
