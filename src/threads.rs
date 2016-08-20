use std::sync::Arc;
use std::thread;

use cmaes::options::CMAESEndConditions;

use utils::Individual;
use cmaes_utils::optimize_network;
use fitness::NNFitnessFunction;

pub fn update_generation<T>(generation: &mut Vec<Individual<T>>,
                            cmaes_options: &Vec<CMAESEndConditions>,
                            cmaes_runs: usize,
                            threads: usize)
    where T: 'static + NNFitnessFunction + Clone + Send + Sync
{
    let mut new_generation = Vec::new();
    let cmaes_options = Arc::new(cmaes_options.clone());

    let do_work = Arc::new(move |mut networks: Vec<Individual<T>>| {
        for network in &mut networks {
            optimize_network(network, &cmaes_options.clone(), cmaes_runs)
        }

        networks
    });

    // Thread setup
    if threads > 0 {
        let mut per_thread = vec![0; threads];
        
        let mut t = 0;
        for _ in 0..generation.len() {
            per_thread[t] += 1;

            t = if t == threads - 1 {
                0
            } else {
                t + 1
            };
        }

        let mut sum = 0;

        let mut handles = Vec::new();

        for t in 0..threads {
            let networks = generation[sum..t + per_thread[t]].to_vec().clone();
            let do_work = do_work.clone();

            handles.push(thread::spawn(move || {
                do_work((*networks).to_vec())
            }));

            sum += per_thread[t];
        }

        for h in handles {
            match h.join() {
                Ok(v) => new_generation.extend_from_slice(&v),
                Err(..) => panic!("Panic while running CMA-ES")
            }
        }

        *generation = new_generation;
    } else {
        *generation = do_work(generation.clone());
    }
}
