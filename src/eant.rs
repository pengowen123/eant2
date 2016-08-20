use std::sync::Arc;

use cge::Network;

use cmaes_utils::*;
use generation::initialize_generation;
use mutation::mutate;
use options::EANT2Options;
use select::select;
use fitness::NNFitnessFunction;

/// Returns a neural network with as high a fitness as possible, along with the network's fitness.
/// See the library level documentation for examples.
pub fn eant_loop<T>(object: &T, options: EANT2Options) -> (Network, f64)
	where T: 'static + NNFitnessFunction + Clone + Send + Sync
{
    // Allow user to pass a neural network as an argument to generate the initial population

    // Extract end conditions from the options, using the CMAESOptions methods to take advantage of
    // the checks they make
    // TODO: After CMA-ES todo is complete set thread count to 0
    let cmaes_options = get_cmaes_options(options.cmaes_end_conditions);
    let cmaes_options = cmaes_options.end_conditions;

    // Extract other options
    let threads = options.threads;
    let fitness_threshold = options.fitness_threshold;
    let threshold = options.similar_fitness;
    let max_generations = options.max_generations;
    let inputs = options.inputs;
    let outputs = options.outputs;
    let cmaes_runs = options.cmaes_runs;
    let population_size = options.population_size;
    let offspring_count = options.offspring_count;
    let print = options.print_option;
    let transfer_function = options.transfer_function;

    // Wrap object in an Arc for sharing across threads
    let object = Arc::new(object.clone());

    // Loop counter
    let mut g = 0;

    // Initialize generation
    // One neuron per output, each connected to approximately 50% of inputs (exact number is
    // random)

    let mut generation = initialize_generation(population_size,
                                               offspring_count,
                                               inputs,
                                               outputs,
											   transfer_function,
                                               object);

    // Everything from here on is part of a loop
    // EANT2 repeatedly runs these operations until the end conditions are met (minimum fitness or
    // maximum generations)

    loop {
        // Get fitness of network topologies by optimizing the parameters of each individual with
        // CMA-ES to get their maximum potential
        // This stage makes up for nearly all the running time of the algorithm, sometimes taking
        // hours or days
        // The good news is that EANT2 will only need 20 generations or so at most to find a solution,
        // even on complicated tasks
        // TODO: Multithread here (each thread gets a certain number of networks)
        // Wrap generation in an Arc, and use *generation.make_mut() to modify fitness and network
        // fields

        if print {
            println!("Beginning EANT2 generation {}", g + 1);
        }

        for network in &mut generation {
            optimize_network(network, &cmaes_options, cmaes_runs);
        }

        // Select individuals to go on to the next generation
        // This is necessary because there will be too many individuals in the population after
        // generating new ones

        generation = select(population_size, generation, threshold);
        generation.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).expect("12"));

        // Test end conditions, and if they are met, break the loop

        println!("{:?}", true);
        let best = generation[0].fitness;
        println!("{:?}", false);

        if best <= fitness_threshold || g + 1 >= max_generations {
            let solution = generation[0].clone();

            if print {
                println!("EANT2 terminated in {} generations", g + 1);
                println!("Solution found with size {} and {} fitness", solution.network.size + 1, solution.fitness);
            }

            return (solution.network, best)
        }

        // Add new individuals to the population by mutating the existing ones

        let mut new_individuals = Vec::new();

        for individual in &generation {
            for _ in 0..offspring_count {
                let mut new = individual.clone();

                mutate(&mut new);

                new_individuals.push(new);
            }
        }

        //use this for testing
        //individuals should stay at a small size as long as population size and offspring count
        //are reasonable
        for (a, i) in generation.iter().enumerate() {
            println!("size of individual {}: {}", a, i.network.size + 1);
        }

        // if finished, polish the code and docs up, then multithread it
        // finish todo in cmaes and optimization todo in cge
        // then publish

        generation.extend_from_slice(&new_individuals);

        // Increment gene ages

        for i in &mut generation {
            for age in &mut i.ages {
                *age += 1;
            }
        }

        // Increment loop counter

        g += 1;
    }
}
