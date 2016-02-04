// CMA-ES is used to optimize the connection weights of a generation of neural networks
// Implemented based on example code at:
// en.wikipedia.org/wiki/CMA-ES

extern crate la;

use la::{Matrix, EigenDecomposition};

use cge::functions::*;
use cge::network::Network;
use cmaes::fitness::FitnessFunction;
use cmaes::network::NetworkParameters;
use cmaes::mvn::sample_mvn;

pub fn cmaes_loop<T>(_: T, network: &mut Network, threads: u8) -> Vec<f64>
    where T: FitnessFunction
{
    // A 2x2 matrix has a less than 1/18e19^3 chance of being singular
    // If the program crashes, consider yourself lucky
    let d = network.genome.len();
    let n = d as f64;
    let sample_size = 4.0 + (3.0 * n.ln()).floor();
    let parents = (sample_size / 2.0).floor() as i32;
    let sample_size = sample_size as i32;

    let mut generation = vec![NetworkParameters::new(vec![]); sample_size as usize];
    let mut covariance_matrix: Matrix<f64> = Matrix::id(d, d);
    let mut mean_vector = vec![0.0; d];
    let mut step_size = 3.0;
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

    loop {
        for i in 0..sample_size as usize {
            generation[i] = NetworkParameters::new(sample_mvn(step_size, &mean_vector, &covariance_matrix));
            network.set_parameters(&generation[i].parameters);
            generation[i].fitness = T::get_fitness(network);
            network.clear_genome();
        }

        generation.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        let mut mean = vec![0.0; d];

        for (i, parent) in generation[0..parents as usize].iter().enumerate() {
            mean = add_vec(&mean, &mul_vec(&parent.parameters, weights[i]));
        }

        let mean_vector_old = mean_vector.clone();
        mean_vector = mean;
        let diff = sub_vec(&mean_vector, &mean_vector_old);

        inv_sqrt_cov = Matrix::new(2, 2, vec![0.3, 0.0, 0.0, 0.3]);
        let a_ = mul_vec(path_s.get_data(), 1.0 - cs);
        let b_ = (cs * (2.0 - cs) * variance_eff).sqrt();
        let c_ = inv_sqrt_cov.scale(b_);
        let e_ = (&c_ * Matrix::vector(diff.clone())).scale(1.0 / step_size);
        let f_ = add_vec(&a_, e_.get_data());
        path_s = Matrix::vector(f_);

        let hs = if (magnitude(path_s.get_data())
                    / (1.0 - (1.0 - cs)).sqrt())
                    .powf(2.0 * g as f64 / sample_size as f64)
                    / expectation < 1.4 + 2.0 / n + 1.0
                    { 1.0 } else { 0.0 };

        let a_ = mul_vec(path_c.get_data(), 1.0 - cc);
        let b_ = hs * (cc * (2.0 - cc) * variance_eff).sqrt();
        let d_ = mul_vec(&diff, b_);
        let e_ = div_vec(&d_, step_size);
        let f_ = add_vec(&a_, &e_);
        path_c = Matrix::vector(f_);

        let mut artmp = transpose(&Matrix::new(d, parents as usize, concat(&generation[0..parents as usize].iter().map(|p| {
            p.parameters.clone()
        }).collect())));
        
        let artmp2 = transpose(&Matrix::new(d, parents as usize, concat(&(0..parents as usize).map(|_| {
            mean_vector_old.clone()
        }).collect()))).scale(1.0 / step_size);

        artmp = artmp - artmp2;

        println!("{:?}", path_c);

        let a_ = covariance_matrix.scale(1.0 - c1 - cmu);
        let b_ = (&path_c * transpose(&path_c) + covariance_matrix.scale(((1.0 - hs) * cc * (2.0 - cc)))).scale(c1);
        let c_ = transpose(&artmp.scale(cmu)) * Matrix::diag(weights.clone()) * &artmp;
        covariance_matrix = a_ + b_ + c_;

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
            let mut b = e.get_v().clone();

            normalize_eigenvectors(&mut b);

            let mut d_ = e.get_v().clone();

            for i in 0..d {
                let cell = d_.get(i, i);
                d_.set(i, i, cell);
            };

            inv_sqrt_cov = &b * d_ * transpose(&b);
        }

        g += 1;

        let best = generation[0].fitness;
        if best < 0.01 {
            println!("{:?}, {}", best, g);
            break;
        }
    }

    generation[0].parameters.clone()
}
