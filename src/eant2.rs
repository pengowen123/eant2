use std::sync::Arc;

use cge::{Network, Activation};
use cmaes::{restart::{RestartStrategy}};
use typed_builder::TypedBuilder;

use crate::{FitnessFunction, generation::Generation, select, mutation::mutate, mutation_probabilities::MutationProbabilities};

const DEFAULT_SEED:                     Option<Network>       = None;
const DEFAULT_ACTIVATION:               Activation            = Activation::Sigmoid;
const DEFAULT_POPULATION_SIZE:          usize                 = 10;
const DEFAULT_OFFSPRING_COUNT:          usize                 = 4;
const DEFAULT_SIMILAR_FITNESS:          f64                   = 0.15;
const DEFAULT_MAX_GENERATIONS:          usize                 = 30;
const DEFAULT_TERMINATING_FITNESS:      f64                   = 0.0;
const DEFAULT_MUTATION_PROBABILITIES:   MutationProbabilities = MutationProbabilities::assemble((3, 8, 1, 3));

// TODO: examine these defaults, idk! This is all tied up in exactly how you optionally specify termination conditions
//       and restarting technique.
const DEFAULT_EXPLOITATION_TERMINATION: Termination = Termination { fitness: 0.0, generations: 30 };
const DEFAULT_EXPLORATION_TERMINATION:  Termination = Termination { fitness: 0.0, generations: 30 };
const DEFAULT_TERMINATION_CRITERIA:     TerminationCriteria = TerminationCriteria {
  exploitation: DEFAULT_EXPLOITATION_TERMINATION,
  exploration:  DEFAULT_EXPLORATION_TERMINATION,
};

#[derive(TypedBuilder)]
/// - When should the EANT2 algorithm terminate?
/// - - For each EANT2 iteration, when should the CMA-ES algorithm terminate?
pub struct Termination {
  #[builder(default = DEFAULT_TERMINATING_FITNESS)]
  /// Fitness crossing this threshold will cause the algorithm to terminate (think: goal reached).
  /// - Defaults to `0.0`.
  pub fitness: f64,
  
  #[builder(default = DEFAULT_MAX_GENERATIONS)]
  /// Limit on the number of generations (iterations of the EANT2 algorithm). 
  /// - Defaults to `30`.
  pub generations: usize
}

/// - When should the EANT2 algorithm terminate?
/// - - For each EANT2 iteration, when should the CMA-ES algorithm terminate?
#[derive(TypedBuilder)]
pub struct TerminationCriteria {
  #[builder(default = DEFAULT_EXPLORATION_TERMINATION)]
  /// The termination criteria for the EANT2 algorithm (think: outer loop, structural evolution).
  pub exploration:  Termination,

  /// The termination criteria for the CMA-ES algorithm (think: inner loop, parameter evolution).
  #[builder(default = DEFAULT_EXPLOITATION_TERMINATION)]
  pub exploitation: Termination
}

/// The EANT2 algorithm.
/// 
/// # Example
/// 
/// ```rust
/// use eant2::*;
/// 
/// let eant = EANT2::builder()
///   .inputs(10)           // required.
///   .outputs(3)           // required.
///   .terminate(
///     TerminationCriteria::builder()
///       .fitness(0.15)    // Either terminate when fitness reaches 0.15,
///       .generations(11)  // or terminate after 11 generations.
///       .build()
///    )
///   .activation(Activation::Relu)
///   .build();
/// 
/// // ...
/// ```
#[derive(TypedBuilder)]
#[builder(doc)]
pub struct EANT2 {
  /// Network input size
  #[builder(setter(doc = "Network input size. Required."))]
  pub inputs: usize,

  /// Network output size
  #[builder(setter(doc = "Network output size. Required."))]
  pub outputs: usize,

  /// CMA-ES parameter optimization (exploitation) restart strategy. Defaults to `RestartStrategy::BIPOP`.
  #[builder(
    default_code = "RestartStrategy::BIPOP(Default::default())",
    setter(doc = "CMA-ES parameter optimization (exploitation) restart strategy. Defaults to `RestartStrategy::BIPOP`."))]
  pub restart: RestartStrategy,

  #[builder(
    default = DEFAULT_POPULATION_SIZE, 
    setter(doc = "Sets the population size. Increasing this option may produce higher quality neural
                  networks, but will increase the time needed to find a solution."))]
  pub population: usize,

  #[builder(
    default = DEFAULT_OFFSPRING_COUNT, 
    setter(doc = "Sets the offspring count (how many offspring each individual spawns). Increasing this
                  option may produce higher quality neural networks, but will increase the time needed to
                  find a solution."))]
  pub offspring: usize,

  #[builder(default = DEFAULT_TERMINATION_CRITERIA, setter(doc = "Algorithm termination conditions"))]
  pub terminate: TerminationCriteria,

  #[builder(
    default = DEFAULT_SIMILAR_FITNESS, 
    setter(doc="Sets the threshold for deciding whether two neural networks have a similar fitness.
                Increasing this option will make the algorithm more aggresively prefer smaller
                neural networks. Decreasing it will do the opposite, allowing larger individuals to stay in
                the population. It is recommended to set this option higher if a small neural network is
                preferred. The downside is it will take slightly longer to find a solution, due to more
                higher fitness neural networks being discarded."))]
  pub similar_fitness: f64,

  #[builder(setter(strip_bool, doc = "Print information throughout the optimization process."))]
  pub print: bool,

  #[builder(
    default = DEFAULT_MUTATION_PROBABILITIES, 
    setter(doc = "Sets the weight for choosing each mutation. In general,
                  the chance for connection mutations should be higher than the chance for a neuron mutation.
                  If a minimal network is desired, set the bias, neuron and connection addition probabilities low,
                  and the connection removal probability high. For complex problems where network size isn't an
                  issue, high connection addition probability is a good idea."))]
  pub mutation_probabilities: MutationProbabilities,

  #[builder(default = DEFAULT_SEED, setter(strip_option))]
  pub seed: Option<Network>,

  #[builder(default = DEFAULT_ACTIVATION, setter(doc = "Activation function the network uses."))]
  pub activation: Activation
}

impl EANT2 {
  /// Run the optimization algorithm until termination conditions are met, yielding the best network and its fitness.
  pub fn run<T>(&self, object: &T) -> (Network, f64) where T: 'static + FitnessFunction + Clone + Send + Sync {
    let object = Arc::new(object.clone());
    let mut g = 0;
    let mut generation = Generation::initialize(&self, object);

    loop {
      // 1. Get fitness of network topologies by optimizing the parameters of each individual with CMA-ES to get their maximum potential.
      //    - This stage makes up for nearly all the running time of the algorithm, sometimes taking hours or days.
      //    - The good news is that EANT2 will only need 20 generations or so at most to find a solution, even on complicated tasks.
      if self.print { println!("Beginning EANT2 generation {}", g + 1); }
      generation.update_generation(&self);

      // 2. Select individuals to go on to the next generation
      generation = select::select(self.population, &generation.individuals[..], self.similar_fitness);
      generation.individuals.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

      let best = &generation.individuals[0];
      
      // 3. Check EANT2 termination conditions
      if best.fitness <= self.terminate.exploration.fitness || g + 1 >= self.terminate.exploration.generations {
        if self.print {
          println!("EANT2 terminated in {} generations", g + 1);
          println!("Solution found with size {} and {} fitness", best.network.size + 1, best.fitness);
        }

        return (best.network.clone(), best.fitness);
      }

      if self.print { println!("Current best fitness: {}", best.fitness); }

      // 4. Add new individuals to the population by mutating the existing ones
      // TODO: consider parallelizing this step
      let mut new_individuals = Vec::with_capacity(self.offspring * self.population);
      for individual in generation.individuals.iter() {
        for _ in 0..self.offspring {
          let mut new = individual.clone();
          mutate(&mut new, self.mutation_probabilities);
          
          // increment the gene ages here. We don't want to revisit memory later when we could do the job now (cpu cache).
          new.ages.iter_mut().for_each(|a| *a += 1);

          new_individuals.push(new);
        }
      }

      generation.individuals.extend_from_slice(&new_individuals);
      g += 1;
    }
  }
}

#[cfg(test)]
mod test {
    // use super::EANT2;

  #[test]
  fn test_builder() {
    // let x = EANT2::builder()
    //   .inputs(10)
    //   .outputs(3)
    //   .build();
  }
}