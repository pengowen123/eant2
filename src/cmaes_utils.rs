use cmaes::*;
use cmaes::objective_function::Scale;
use cmaes::restart::{Restarter, RestartOptions};
use crate::eant2::EANT2;
use crate::utils::Individual;
use crate::FitnessFunction;

/// The structural __exploitation__ phase of the algorithm,
/// where the parameters of each neural topology (created in the structural __exploration__ phase) are optimized.
// TODO: amortize allocations!
// TODO: Nice API design to specify termination conditions + restart strategies
pub fn optimize_network<T>(individual: &mut Individual<T>, options: &EANT2)
  where T: 'static + FitnessFunction + Clone + Send + Sync {
  
  // g' = 1 / (1 + (g^2))
  // used to restrict the search space weights as the gene ages (this is supposed to encourage better convergence)
  let gene_deviations: Vec<f64> = 
    individual
      .ages
      .iter()
      .map(|&g| g * g) // integer math more performant
      .map(|g| 1 + g) // integer math more performant
      .map(|g| 1.0 / g as f64) // now we cannot use integer math, use the floating point instructions
      .collect();
      
  // TODO: find an ergonomic way to optionally specify termination conditions, restarter, etc. while also capturing (statically in the type system!)
  //       that there is a minimum amount of information that must be provided to prevent infinite looping.
  let best = {
    // TODO: amortize allocation
    let initial_mean = DVector::from(individual.network.genome.iter().map(|g| g.weight).collect::<Vec<f64>>());
    let initial_step_size = 0.3;

    // TODO: avoid these clones if possible.
    let parameter_count = individual.network.genome.len();
    let scaled = Scale::new(|x: &DVector<f64>| individual.evaluate(&(x + &initial_mean)), gene_deviations.clone());

    // run the CMA-ES optimization pass
    let restarter = 
      Restarter::new(
        RestartOptions::new(
          parameter_count, 
          -1.0..=1.0,
          options.restart.clone()
        )
        .mode(Mode::Minimize)
        .enable_printing(options.print)
        .fun_target(options.terminate.exploitation.fitness)
        .max_generations_per_run(options.terminate.exploitation.generations)
      )
      .unwrap();

    restarter.clone().run_with_reuse(scaled).best.unwrap()
  };
  
  // extract the best parameters
  let best_parameters = &best.point;

  // commit to the new network parameters (the returned value does not have parameter scaling applied, so we do that here!)
  individual
    .network
    .genome
    .iter_mut()
    .zip(best_parameters.iter())
    .zip(gene_deviations.iter())
    .for_each(|((gene, &new_weight), &scale)| gene.weight = new_weight * scale);

  // update the fitness of the network (with its new parameters)
  individual.fitness = best.value;
}

fn build_cmaes_options(
  initial_mean: Vec<f64>, 
  initial_step_size: f64, 
  options: &EANT2
) -> CMAESOptions {
  CMAESOptions::new(initial_mean, initial_step_size)
    .fun_target(options.terminate.exploitation.fitness)
    .max_generations(options.terminate.exploitation.generations)
    .mode(Mode::Minimize)
}