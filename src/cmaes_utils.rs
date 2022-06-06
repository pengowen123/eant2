use crate::eant2::EANT2;
use crate::utils::Individual;
use crate::FitnessFunction;
use cmaes::objective_function::Scale;
use cmaes::restart::{RestartOptions, Restarter};
use cmaes::*;

/// The structural __exploitation__ phase of the algorithm,
/// where the parameters of each neural topology (created in the structural __exploration__ phase) are optimized.
// TODO: amortize allocations!
// TODO: Nice API design to specify termination conditions + restart strategies
pub fn optimize_network<T>(individual: &mut Individual<T>, options: &EANT2)
where
    T: 'static + FitnessFunction + Clone + Send + Sync,
{
    // g' = 1 / (1 + (g^2))
    // used to restrict the search space weights as the gene ages (this is supposed to encourage better convergence)
    let gene_deviations: Vec<f64> = individual
    .ages
    .iter()
    .map(|&g| g * g) // integer math more performant
    .map(|g| 1 + g) // integer math more performant
    .map(|g| 1.0 / g as f64) // now we cannot use integer math, use the floating point instructions
    .collect();

    // TODO: amortize allocation
    let initial_mean = DVector::from(individual.network.weights().collect::<Vec<f64>>());

    // TODO: find an ergonomic way to optionally specify termination conditions, restarter, etc. while also capturing (statically in the type system!)
    //       that there is a minimum amount of information that must be provided to prevent infinite looping.
    let best = {
        // TODO: avoid these clones if possible.
        let parameter_count = individual.network.len();
        let scaled = Scale::new(
            |x: &DVector<f64>| individual.evaluate(&(x + &initial_mean)),
            gene_deviations.clone(),
        );

        let mut restart_options = RestartOptions::new(parameter_count, -1.0..=1.0, options.exploitation.restart.clone())
          .mode(Mode::Minimize)                  // minimize the fitness function
          .fun_target(options.exploration.terminate.fitness); // don't optimize beyond the EANT2 fitness

        if let Some(max_gens) = options.exploitation.terminate.generations {
            restart_options = restart_options.max_generations_per_run(max_gens);
        }
        if let Some(max_evals) = options.exploitation.terminate.evaluations {
            restart_options = restart_options.max_function_evals(max_evals);
        }

        // run the CMA-ES optimization pass
        Restarter::new(restart_options)
          .unwrap()
          .run_with_reuse(scaled)
          .best
          .expect("CMA-ES optimization failed, this is likely the result of FitnessFunction returning f64::NAN")
    };

    // extract the best parameters
    let best_parameters = &best.point;

    // commit to the new network parameters (the returned value does not have parameter scaling applied, so we do that here!)
    individual
        .network
        .mut_weights()
        .zip(best_parameters.iter())
        .zip(initial_mean.iter())
        .zip(gene_deviations.iter())
        .for_each(|(((weight, initial), &new_weight), &scale)| {
            *weight = initial + (new_weight * scale)
        });

    // update the fitness of the network (with its new parameters)
    individual.fitness = best.value;
}
