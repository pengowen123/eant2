use cmaes::*;
use cmaes::options::CMAESEndConditions;
use cmaes::options::CMAESEndConditions::*;
use cge::gene::Gene;

use utils::Individual;
use fitness::NNFitnessFunction;

pub fn optimize_network<T>(individual: &mut Individual<T>,
                           cmaes_options: &[CMAESEndConditions],
                           cmaes_runs: usize)
    where T: 'static + NNFitnessFunction + Clone + Send + Sync
{

    let gene_deviations: Vec<f64> = individual.ages.iter().map(|g| {
        1.0 / (1.0 + (*g as f64).powi(2))
    }).collect();

    let mut individual_options = CMAESOptions {
        end_conditions: cmaes_options.to_vec(),
        dimension: individual.network.size + 1,
        initial_step_size: 0.3,
        initial_standard_deviations: gene_deviations,
        initial_mean: individual.network.genome.iter().map(|g| g.weight).collect(),
        threads: 0
    };

    for _ in 0..cmaes_runs {
        let results = cmaes_loop(individual, individual_options.clone()).unwrap();

        individual.network.genome = individual.network.genome.iter().enumerate().map(|(i, g)| {
            Gene {
                weight: results.0[i],
                .. g.clone()
            }
        }).collect();

        individual.fitness = results.1;
        individual_options.initial_mean = individual.network.genome.iter().map(|g| g.weight).collect();
    }
}

pub fn get_cmaes_options(conditions: Vec<CMAESEndConditions>) -> CMAESOptions {
    let mut cmaes_options = CMAESOptions::custom(1);

    for condition in conditions {
        match condition {
            StableGenerations(ref fitness, ref generations) => {
                cmaes_options = cmaes_options.stable_generations(*fitness, *generations);
            },
            FitnessThreshold(ref fitness) => {
                cmaes_options = cmaes_options.fitness_threshold(*fitness);
            },
            MaxGenerations(ref generations) => {
                cmaes_options = cmaes_options.max_generations(*generations);
            },
            MaxEvaluations(ref evaluations) => {
                cmaes_options = cmaes_options.max_evaluations(*evaluations);
            }
        }
    }

    cmaes_options
}
