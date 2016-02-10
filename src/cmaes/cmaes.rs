// Add the end condition Never(g)
// Use a separate function that returns a struct containing a handle
// to a thread and a method to get data from that thread
// The best individual, step size, covariance matrix, and other stuff should be put\
// into that thread every g generations
// This is really similar to a yield keyword
//
// Clean this up; move state variable updates to separate modules if possible

extern crate rand;
extern crate la;

use std::usize;
use std::thread;
use std::sync::{Arc, Mutex};

use la::{Matrix, EigenDecomposition};
use rand::random;
use rand::distributions::{Normal, IndependentSample};

use cge::functions::*;
use cge::network::Network;
use cmaes::fitness::FitnessFunction;
use cmaes::network::NetworkParameters;
use cmaes::condition::CMAESEndConditions;

const MIN_STEP_SIZE: f64 = 1e-290;
const MAX_STEP_SIZE: f64 = 1e290;

pub fn cmaes_loop<T>(_: T,
                     mut network: Network,
                     threads: u8,
                     condition: CMAESEndConditions) -> Vec<f64>

    where T: FitnessFunction
{
    //! The main CMA-ES function. The algorithm minimizes the fitness function, so a lower fitness
    //! represents a better individual.
    //! This documentation is currently not correct, but will be soon.
    //! 
    //! # Examples
    //!
    //! ```
    //! use cmaes::*;
    //!
    //! struct FitnessDummy;
    //! 
    //! impl FitnessFunction for FitnessDummy {
    //!     fn get_fitness(parameters: &[f64]) -> f64 {
    //!         // Calculate fitness here
    //!         
    //!         0.0
    //!     }
    //! }
    //!
    //! let condition = CMAESEndConditions::FitnessTheshold(0.00001);
    //!
    //! let solution = cmaes_loop(FitnessDummy, 1, condition);
    //! ```
    //! 
    //! # Panics
    //!
    //! Panics if the fitness function panics or returns NaN or infinite.

    if threads == 0 {
        panic!("Threads must be at least one");
    }

    // Various numbers; mutable variables are only used as a starting point and
    // are adapted by the algorithm
    let d = network.genome.len();
    let n = d as f64;
    let sample_size = 4.0 + (3.0 * n.ln()).floor();
    let parents = (sample_size / 2.0).floor() as i32;
    let sample_size = sample_size as i32;
    
    let mut generation; 
    let mut covariance_matrix: Matrix<f64> = Matrix::id(d, d);
    let mut eigenvectors = Matrix::id(d, d);
    let mut eigenvalues = Matrix::vector(vec![1.0; d]);
    let mut mean_vector = vec![random(); d];
    let mut step_size = 0.3;
    let mut path_s: Matrix<f64> = Matrix::vector(vec![0.0; d]);
    let mut path_c: Matrix<f64> = Matrix::vector(vec![0.0; d]);
    let mut inv_sqrt_cov: Matrix<f64> = Matrix::id(d, d);
    let mut g = 0;
    let mut eigeneval = 0;
    let mut hs;

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

    // Thread stuff
    let mut per_thread = vec![0; sample_size as usize];
    
    let mut t = 0;
    for _ in 0..sample_size {
        per_thread[t] += 1;
        t = if threads as usize > t { 0 } else { t + 1};
    }
        
    let network = Arc::new(Mutex::new(network));
    let dist = Normal::new(0.0, 1.0);

    // End condition variables
    let mut stable = 0;
    let mut best = 0.0;

    loop {
        // More thread stuff
        generation = Vec::new();
        let vectors = Arc::new(eigenvectors.clone());
        let values = Arc::new(eigenvalues.clone());
        let mean = Arc::new(mean_vector.clone());

        // Create new individuals
        for t in per_thread.clone() {
            let thread_network = network.clone();
            let thread_mean = mean.clone();
            let thread_vectors = vectors.clone();
            let thread_values = values.clone();

            let handle = thread::spawn(move || {
                let mut individuals = Vec::new();

                for _ in 0..t as usize {
                    let random_values;
                    
                    {
                        random_values = vec![dist.ind_sample(&mut rand::thread_rng()); d];
                    }

                    // Sample multivariate normal
                    let parameters = mul_vec_2(&*thread_values.get_data(), &random_values);
                    let parameters = matrix_by_vector(&*thread_vectors, &parameters);
                    let parameters = add_vec(&*thread_mean, &mul_vec(&parameters, step_size));

                    // Get fitness of parameters
                    let mut thread_network = thread_network.lock().unwrap();
                    thread_network.set_parameters(&parameters);
                    let mut individual = NetworkParameters::new(parameters);
                    let fitness = T::get_fitness(&mut *thread_network);
                    individual.fitness = fitness;

                    // Protect from invalid values
                    if fitness.is_nan() || fitness.is_infinite() {
                        panic!("Fitness function returned NaN or infinite");
                    }

                    // Reset network state; will get rid of these calls
                    thread_network.clear_genome();

                    individuals.push(individual);
                }

                individuals
            });

            // User-defined function might panic
            let individuals = match handle.join() {
                Ok(v) => v,
                Err(..) => panic!("Panicked while calling fitness function")
            };

            for item in individuals {
                generation.push(item);
            }
        }
        
        // Increment function evaluations counter
        g += sample_size as usize;

        // Sort generation by fitness; smallest fitness will be first
        generation.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // Update mean vector
        // New mean vector is the average of the parents
        let mut mean = vec![0.0; d];
        for (i, parent) in generation[0..parents as usize].iter().enumerate() {
            mean = add_vec(&mean, &mul_vec(&parent.parameters, weights[i]));
        }

        let mean_vector_old = mean_vector.clone();
        mean_vector = mean;

        // To prevent duplicate code
        let diff = sub_vec(&mean_vector, &mean_vector_old);

        // Update the evolution path for the step size (sigma)
        // Measures how many steps the step size has taken in the same direction
        let a_ = mul_vec(path_s.get_data(), 1.0 - cs);
        let b_ = (cs * (2.0 - cs) * variance_eff).sqrt();
        let c_ = inv_sqrt_cov.scale(b_);
        let e_ = (&c_ * Matrix::vector(diff.clone())).scale(1.0 / step_size);
        let f_ = add_vec(&a_, e_.get_data());
        path_s = Matrix::vector(f_);

        // hs determines whether to do an additional update to the covariance matrix
        let a_ = magnitude(path_s.get_data());
        let b_ = (1.0 - (1.0 - cs).powf(2.0 * g as f64 / sample_size as f64)).sqrt();
        hs = (a_ / b_ / expectation < 1.4 + 2.0 / (n + 1.0)) as i32 as f64;

        // Update the evolution path for the covariance matrix (capital sigma)
        // Measures how many steps the step size has taken in the same direction
        let a_ = mul_vec(path_c.get_data(), 1.0 - cc);
        let b_ = hs * (cc * (2.0 - cc) * variance_eff).sqrt();
        let d_ = mul_vec(&diff, b_);
        let e_ = div_vec(&d_, step_size);
        let f_ = add_vec(&a_, &e_);
        path_c = Matrix::vector(f_);

        // Factor in the values of the individuals
        let mut artmp = transpose(&Matrix::new(parents as usize, d, concat(&generation[0..parents as usize].iter().map(|p| {
            p.parameters.clone()
        }).collect())));
        
        let artmp2 = transpose(&Matrix::new(parents as usize, d, concat(&(0..parents as usize).map(|_| {
            mean_vector_old.clone()
        }).collect())));

        artmp = (artmp - artmp2).scale(1.0 / step_size);
        
        // Update the covariance matrix
        // Determines the shape of the search area
        let a_ = covariance_matrix.scale(1.0 - c1 - cmu);
        let b_ = (&path_c * transpose(&path_c) + covariance_matrix.scale(((1.0 - hs) * cc * (2.0 - cc)))).scale(c1);
        let c_ = &artmp.scale(cmu) * Matrix::diag(weights.clone()) * transpose(&artmp);
        covariance_matrix = a_ + b_ + c_;

        // Update the step size
        // Determines the size of the search area
        // Increased if the length of its evolution path is greater than the expectation of N(0, I)
        step_size = step_size * ((cs / damps) * (magnitude(path_s.get_data()) / expectation - 1.0)).exp();

        // Update the eigenvectors and eigenvalues every so often
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

            let mut new = Vec::new();

            {
                let data = eigenvectors.get_data();

                for i in 0..d {
                    let i = i * d;
                    new.extend_from_slice(&reverse(&data[i..i + d]));
                }
            }

            eigenvectors = Matrix::new(d, d, new);

            eigenvalues = e.get_d().clone();

            eigenvalues = Matrix::vector(reverse(&(0..d).map(|i| {
                eigenvalues[(i, i)].sqrt()
            }).collect::<Vec<f64>>()));
            
            let inverse = Matrix::diag(eigenvalues.get_data().iter().map(|n| {
                n.powi(-1)
            }).collect::<Vec<f64>>());

            inv_sqrt_cov = &eigenvectors * inverse * transpose(&eigenvectors);
        }

        // Test the end conditions
        match condition {
            CMAESEndConditions::Stabilized(t, g_) => {
                if (best - generation[0].fitness).abs() < t {
                    stable += 1;
                }

                if stable >= g_ {
                    break;
                }
            },

            CMAESEndConditions::FitnessThreshold(f) => {
                if generation[0].fitness <= f {
                    break;
                }
            },

            CMAESEndConditions::MaxGenerations(g_) => {
                if g / sample_size as usize >= g_ {
                    break;
                }
            },

            CMAESEndConditions::MaxEvaluations(e) => {
                if g > e {
                    break;
                }
            }
        }

        // To prevent bad things from happening
        if step_size <= MIN_STEP_SIZE ||
           step_size >= MAX_STEP_SIZE ||
           g >= usize::MAX {
            break;
        }

        best = generation[0].fitness;
    }

    // Test if the mean vector is better than the best solution
    let network = network.clone();
    let mut network = network.lock().unwrap();
    network.set_parameters(&mean_vector);
    let fitness = T::get_fitness(&mut *network);
    network.clear_genome();

    network.set_parameters(&generation[0].parameters);
    network.step(&vec![9.5, -4.5], false);
    println!("{:?}", network.step(&vec![0.2, 0.1], false)[0]);

    if fitness < best {
        mean_vector
    } else {
        generation[0].parameters.clone()
    }
}
