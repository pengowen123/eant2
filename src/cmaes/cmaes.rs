// I generally understand this algorithm, and have experience with debugging it
// If you have a problem, feel free to ask

extern crate la;

use std::thread;
use std::sync::{Arc, Mutex};

use la::{Matrix, EigenDecomposition};
use rand::random;

use cge::functions::*;
use cge::network::Network;
use cmaes::fitness::FitnessFunction;
use cmaes::network::NetworkParameters;
use cmaes::mvn::sample_mvn;
use cmaes::condition::CMAESEndConditions;

pub fn cmaes_loop<T>(_: T,
                     mut network: Network,
                     threads: u8,
                     condition: CMAESEndConditions) -> Vec<f64>

    where T: FitnessFunction
{
    // A 2x2 matrix has a less than 1/18e19^3 chance of being singular
    // If the program crashes, consider yourself lucky
    
    if threads == 0 {
        panic!("Threads must be at least one");
    }

    let d = network.genome.len();
    let n = d as f64;
    let sample_size = 4.0 + (3.0 * n.ln()).floor();
    let parents = (sample_size / 2.0).floor() as i32;
    let sample_size = sample_size as i32;
    
    if threads as i32 > sample_size {
        println!("Warning: {} unused threads", threads as i32 - sample_size);
    }

    let mut generation = Vec::new(); 
    let mut covariance_matrix: Matrix<f64> = Matrix::id(d, d);
    let mut eigenvectors = Matrix::id(d, d);
    let mut eigenvalues = Matrix::id(d, d);
    let mut mean_vector = vec![random(); d];
    let mut step_size = 0.3;
    let mut path_s: Matrix<f64> = Matrix::vector(vec![0.0; d]);
    let mut path_c: Matrix<f64> = Matrix::vector(vec![0.0; d]);
    let mut inv_sqrt_cov: Matrix<f64> = Matrix::id(d, d);
    let mut g = 0;
    let mut eigeneval = 0;

    let weights = (0..parents).map(|i: i32| {
        (parents as f64 + 1.0 / 2.0).ln() - ((i as f64 + 1.0).ln())
    }).collect::<Vec<f64>>();

    let sum = sum_vec(&weights);
    let weights = weights.iter().map(|i| { i / sum }).collect::<Vec<f64>>();
    let variance_eff = sum_vec(&weights) / sum_vec(&mul_vec_2(&weights, &weights));
    let cc = (4.0 + variance_eff / n as f64) / (n + 4.0 + 2.0 * variance_eff / n);
    let cs = (variance_eff + 2.0) / (n + variance_eff + 5.0);
    let c1 = 2.0 / ((n + 1.3).powi(2) + variance_eff);
    let cmu = {
        let a = 1.0 - c1;
        let b = 2.0 * (variance_eff - 2.0 + 1.0 / variance_eff) / ((n + 2.0).powi(2) + variance_eff);
        if a <= b { a } else { b }
    };
    let damps = 1.0 + cs + 2.0 * {
        let result = ((variance_eff - 1.0) / (n + 1.0)).sqrt() - 1.0;
        if 0.0 > result { 0.0 } else { result }
    };
    let expectation = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n.powi(2)));
    
    // Clear network genome before starting
    network.clear_genome();
    // Number of individuals assigned to each thread
    let mut per_thread = vec![0; sample_size as usize];
    
    let mut t = 0;
    for _ in 0..sample_size {
        per_thread[t] += 1;
        t = if threads as usize > t { 0 } else { t + 1};
    }
        
    // Wrap network in Arc and Mutex for accessing across threads
    let network = Arc::new(Mutex::new(network));

    let mut end = false;
    let mut stable = 0;
    let mut best = 0.0;

    while !end {
        generation = Vec::new();
        let mean = Arc::new(mean_vector.clone());
        let vectors = Arc::new(eigenvectors.clone());
        let values = Arc::new(eigenvalues.clone());

        for t in per_thread.clone() {
            let thread_network = network.clone();
            let thread_mean = mean.clone();
            let thread_vectors = vectors.clone();
            let thread_values = values.clone();

            let handle = thread::spawn(move || {
                let mut individuals = Vec::new();

                for _ in 0..t as usize {
                    let parameters = sample_mvn(step_size,
                                                &thread_mean,
                                                &thread_vectors,
                                                &thread_values);

                    let mut thread_network = thread_network.lock().unwrap();
                    thread_network.set_parameters(&parameters);

                    let mut individual = NetworkParameters::new(parameters);
                    individual.fitness = T::get_fitness(&mut *thread_network);

                    if individual.fitness.is_nan() {
                        println!("Warning: Fitness function returned NaN");
                    }

                    if individual.fitness.is_infinite() {
                        println!("Warning: Fitness function returned infinity");
                    }

                    thread_network.clear_genome();

                    individuals.push(individual);
                }

                individuals
            });

            let individuals = match handle.join() {
                Ok(v) => v,
                Err(e) => panic!("Error while calling fitness function: {:?}", e)
            };

            for item in individuals {
                generation.push(item);
            }
        }

        g += sample_size as usize;

        generation.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        let mut mean = vec![0.0; d];

        for (i, parent) in generation[0..parents as usize].iter().enumerate() {
            mean = add_vec(&mean, &mul_vec(&parent.parameters, weights[i]));
        }

        let mean_vector_old = mean_vector.clone();
        mean_vector = mean;
        let diff = sub_vec(&mean_vector, &mean_vector_old);

        let a_ = mul_vec(path_s.get_data(), 1.0 - cs);
        let b_ = (cs * (2.0 - cs) * variance_eff).sqrt();
        let c_ = inv_sqrt_cov.scale(b_);
        let e_ = (&c_ * Matrix::vector(diff.clone())).scale(1.0 / step_size);
        let f_ = add_vec(&a_, e_.get_data());
        path_s = Matrix::vector(f_);

        let a_ = magnitude(path_s.get_data());
        let b_ = (1.0 - (1.0 - cs).powf(2.0 * g as f64 / sample_size as f64)).sqrt();
        let hs = (a_ / b_ / expectation < 1.4 + 2.0 / (n + 1.0)) as i32 as f64;
        
        let a_ = mul_vec(path_c.get_data(), 1.0 - cc);
        let b_ = hs * (cc * (2.0 - cc) * variance_eff).sqrt();
        let d_ = mul_vec(&diff, b_);
        let e_ = div_vec(&d_, step_size);
        let f_ = add_vec(&a_, &e_);
        path_c = Matrix::vector(f_);

        let mut artmp = transpose(&Matrix::new(parents as usize, d, concat(&generation[0..parents as usize].iter().map(|p| {
            p.parameters.clone()
        }).collect())));
        
        let artmp2 = transpose(&Matrix::new(parents as usize, d, concat(&(0..parents as usize).map(|_| {
            mean_vector_old.clone()
        }).collect())));

        artmp = (artmp - artmp2).scale(1.0 / step_size);

        let mut covariance_matrix = Matrix::new(2, 2, vec![0.8f64, 0.3, 0.3, 0.8]);
        let artmp = Matrix::new(2, 3, vec![0.5, 0.4, 0.2, 0.8f64, 0.6, 0.3]);
        let path_c = Matrix::vector(vec![0.5, 0.4f64]);
        let hs = 0.0;
        let cc = 0.9;
        let c1 = 0.5;
        let cmu = 1.1;
        let weights = vec![0.5, 0.3, 0.2];

        let a_ = covariance_matrix.scale(1.0 - c1 - cmu);
        let b_ = (&path_c * transpose(&path_c) + covariance_matrix.scale(((1.0 - hs) * cc * (2.0 - cc)))).scale(c1);
        let c_ = &artmp.scale(cmu) * Matrix::diag(weights.clone()) * transpose(&artmp);
        covariance_matrix = a_ + b_ + c_;
        println!("{:?}", covariance_matrix);

        step_size = step_size * ((cs / damps) * (magnitude(path_s.get_data()) / expectation - 1.0)).exp();

        if (g - eigeneval) as f64 > sample_size as f64 / (c1 + cmu) / n / 10.0 {
            eigeneval = g;

            for y in 0..d {
                for x in y + 1..d {
                    let cell = covariance_matrix.get(y, x);
                    covariance_matrix.set(x, y, cell);
                }
            }

            let e = EigenDecomposition::new(&covariance_matrix);
            eigenvectors = e.get_v().clone();
            eigenvalues = e.get_d().clone();

            for i in 0..d {
                let cell = eigenvalues.get(i, i);
                eigenvalues.set(i, i, cell.powf(-0.5));
            };

            inv_sqrt_cov = &eigenvectors * &eigenvalues * transpose(&eigenvectors);
        }

        match condition {
            CMAESEndConditions::Stabilized(t, g) => {
                if (best - generation[0].fitness).abs() < t {
                    stable += 1;
                }

                if stable > g {
                    end = true;
                }
            },

            CMAESEndConditions::FitnessThreshold(f) => {
                if generation[0].fitness < f {
                    end = true;
                }
            },

            CMAESEndConditions::MaxGenerations(g_) => {
                if g / sample_size as usize > g_ {
                    end = true;
                }
            }
        }

        best = generation[0].fitness;
    }

    let network = network.clone();
    let mut network = network.lock().unwrap();
    network.set_parameters(&mean_vector);

    if T::get_fitness(&mut *network) > best {
        mean_vector
    } else {
        generation[0].parameters.clone()
    }
}
