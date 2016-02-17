// TODO: Multithread this (important)
//
// Mutation operators
// read paper for more info
//
// Add/Remove connection: Add/Delete gene and increment/decrement input count of the previous
// neuron
//
// Add bias: Add bias gene which stores a weight, and has a value stored in a const
//
// Add subnetwork: Add neuron gene and connect it to approx. 50% of inputs (use forward jumper
// genes)
//
// Add wrapper structs to allow a vector of only Inputs and Neurons, Forward and Recurrent Jumpers,
// etc. to easily index things that share a property for mutations. Either this or several vectors
// of indexes of genes available for certain operations.

use std::sync::Arc;

use cge::Network;
use cge::gene::Gene;
use cmaes::*;
use cmaes::options::CMAESEndConditions::*;
use rand::random;

use utils::{Individual, GeneAge};
use options::EANT2Options;
use mutationops::Mutation;
use select::select;
use fitness::NNFitnessFunction;

/// Start value for ln(x) for determining the penalty to initial search radius of gene parameters,
/// where x is the gene's age.
pub const GENE_AGE_LOG_START: f64 = 2.72;

/// Returns a neural network with as high a fitness as possible, along with the network's fitness.
/// Add more docs and examples after completion
pub fn eant_loop<T>(object: &T, options: EANT2Options) -> (Network, f64)
	where T: 'static + NNFitnessFunction + Clone + Send + Sync
{
    // Allow user to pass a neural network as an argument to generate the initial population
    // Ensure user input is valid
    
    // to make new generation, add new individuals to the population by mutating every individual
    // optimize all individuals with cmaes
    // use the select function to rank and select which individuals go to the next generation
    // refer to the flow chart in the eant2 paper for more info
    // keep this file as abstract as possible
    // make sure there are no free floating structures
    // protect against non-normal fitness values and in general other values such as generation
    // count

    // Extract end conditions from the options, using the CMAESOptions methods to take advantage of
    // the checks they make
    let mut cmaes_options = CMAESOptions::custom(1);

    for condition in options.cmaes_end_conditions {
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

    let cmaes_options = cmaes_options.end_conditions;

    // Extract other options
    let threads = options.threads;
    let fitness_threshold = options.fitness_threshold;
    let max_generations = options.max_generations;
    let inputs = options.inputs;
    let outputs = options.outputs;
    let cmaes_runs = options.cmaes_runs;
    let population_size = options.population_size;
    let offspring_count = options.offspring_count;

    // Wrap object in an Arc for sharing across threads
    let object = Arc::new(object.clone());

    // ID of the next neuron
    let mut neuron_id = outputs;

    // Loop counter
    
    let mut g = 0;

    // Initialize generation
    // One neuron per output, each connected to approximately 50% of inputs (exact number is
    // random)
    
    let mut generation = Vec::new();

    for _ in 0..population_size * (offspring_count + 1) {
        let mut network = Network {
            size: 0,
            genome: Vec::new()
        };

        for i in 0..outputs {
            network.add_subnetwork(i, 0, inputs)
        }

        network.size = network.genome.len() - 1;

        generation.push(Individual::new(network, object.clone()));
    }

    // Everything from here on is part of a loop
    // EANT2 repeatedly runs these operations until the end conditions are met (minimum fitness or
    // maximum generations)

    loop {
        // Get fitness of network topologies by optimizing the parameters of each individual with
        // CMA-ES to get their maximum potential
        // This stage takes the longest, potentially hours
        // The good new is that EANT2 will only need 100 generations at most to find a solution,
        // even on complicated tasks
        // TODO: Multithread here (each thread gets a certain number of networks)
        // Wrap generation in an Arc, and use *generation.make_mut() to modify fitness and network
        // fields

        for network in &mut generation {
            let gene_deviations = network.genes.iter().map(|g| {
                1.0 / (g.age as f64 + GENE_AGE_LOG_START).ln().powi(2)
            }).collect();

            // Delete this
            println!("size: {:?}", network.network.size + 1);

            let individual_options = CMAESOptions {
                end_conditions: cmaes_options.clone(),
                dimension: network.network.size + 1,
                initial_step_size: 0.3,
                initial_standard_deviations: gene_deviations,
                initial_mean: network.network.genome.iter().map(|g| g.weight).collect(),
                threads: 1
            };

            let mut results = vec![cmaes_loop(&*network, individual_options.clone()).unwrap(); cmaes_runs];
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let best = results[0].0.clone();

            (*network).network.genome = (*network).network.genome.iter().enumerate().map(|(i, g)| {
                Gene {
                    weight: best[i],
                    .. g.clone()
                }
            }).collect();

            (*network).fitness = results[0].1;
        }

        // Select individuals to go on to the next generation
        // This is necessary because there will be too many individuals in the population after
        // generating new ones
        // This stage does not take very long
        
        generation = select(population_size, generation);

        // Test end conditions, and if they are met, break the loop

        let best = generation[0].fitness;

        if best <= fitness_threshold || g >= max_generations {
            return (generation[0].network.clone(), best)
        }

        // Add new individuals to the population by mutating the existing ones
        
        // continue here, use random() and match statements to select mutations
        // guarantee no invalid mutations are made (might be hard, but is necessary)

        // Increment loop counter
        
        g += 1;
    }
}
