use std::sync::Arc;
use cge::{Network, Activation};
use cmaes::{restart::{RestartStrategy}};
use typed_builder::TypedBuilder;
use crate::{FitnessFunction, generation::Generation, select, mutation::mutate, mutation_probabilities::MutationSampler};

const DEFAULT_ACTIVATION:          Activation       = Activation::Sigmoid;
const DEFAULT_POPULATION_SIZE:     usize            = 10;
const DEFAULT_OFFSPRING_COUNT:     usize            = 4;
const DEFAULT_SIMILAR_FITNESS:     f64              = 0.15;
const DEFAULT_MAX_GENERATIONS:     usize            = 30;
const DEFAULT_TERMINATING_FITNESS: f64              = 0.0;
const DEFAULT_EANT2_TERMINATION:   EANT2Termination = EANT2Termination { fitness: 0.0, generations: 30 };

/// - When should the EANT2 algorithm terminate?
/// - - For each EANT2 iteration, when should the CMA-ES algorithm terminate?
#[derive(TypedBuilder)]
pub struct EANT2Termination {
  #[builder(default = DEFAULT_TERMINATING_FITNESS)]
  /// Fitness crossing this threshold will cause the algorithm to terminate (think: goal reached).
  /// - Defaults to `0.0`.
  pub fitness: f64,
  
  #[builder(default = DEFAULT_MAX_GENERATIONS)]
  /// Limit on the number of generations (iterations of the EANT2 algorithm). 
  /// - Defaults to `30`.
  pub generations: usize
}

#[derive(TypedBuilder)]
pub struct Exploration {
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

  #[builder(
    default = DEFAULT_SIMILAR_FITNESS, 
    setter(doc= "Sets the threshold for deciding whether two neural networks have a similar fitness.
                 Increasing this option will make the algorithm more aggresively prefer smaller
                 neural networks. Decreasing it will do the opposite, allowing larger individuals to stay in
                 the population. It is recommended to set this option higher if a small neural network is
                 preferred. The downside is it will take slightly longer to find a solution, due to more
                 higher fitness neural networks being discarded."))]
  pub similar_fitness: f64,

  #[builder(
    default_code = "MutationSampler::default()", 
    setter(doc = "Sets the sampler which chooses each mutation.  Build a `MutationSampler` with `MutationProbabilities`."))]
  pub mutation_probabilities: MutationSampler,

  #[builder(default = DEFAULT_EANT2_TERMINATION, setter(doc = "Termination conditions (target fitness, max generations)"))]
  pub terminate: EANT2Termination,
}

#[derive(TypedBuilder)]
pub struct CMAESTermination {
  #[builder(default = None, setter(strip_option, doc = "Default is to use the fitness from `EANT2Termination` options, which defaults to `0.0`."))]
  pub fitness:     Option<f64>,

  #[builder(default = None, setter(strip_option, doc = "Default is no CMA-ES generation limit."))]
  pub generations: Option<usize>
}

#[derive(TypedBuilder)]
pub struct Exploitation {
  /// CMA-ES parameter optimization restart strategy. Defaults to `RestartStrategy::BIPOP`.
  #[builder(
    default_code = "RestartStrategy::BIPOP(Default::default())",
    setter(doc = "CMA-ES parameter optimization (exploitation) restart strategy. Defaults to `RestartStrategy::BIPOP`."))]
  pub restart: RestartStrategy,

  #[builder(
    default_code = "CMAESTermination::builder().build()",
    setter(doc = "CMA-ES parameter optimization (exploitation) termination conditions (target fitness, max generations)."))]
  pub terminate: CMAESTermination
}

/// The EANT2 algorithm.
/// 
/// ```text
/// ┌EANT2─────────────────────┐
/// │     ┌──────────────────┐ │
/// │     │ Minimal Networks │ │
/// │     └─────────────────┬┘ │
/// │ ┌CMA-ES───────────────▼┐ │
/// │ │Parameter Optimization│ │
/// │ └▲────────────────────┬┘ │
/// │  │ ┌──────────────────▼┐ │
/// │  │ │ Network Selection │ │
/// │  │ └──────────────────┬┘ │
/// │ ┌┴────────────────────▼┐ │
/// │ │  Structural Mutation │ │
/// │ └──────────────────────┘ │
/// └──────────────────────────┘
/// ```
/// 
/// # Minimal Example
/// - Most options have good default values, and exist for flexibility.
/// - See the `Advanced Example` to see all the options that are normally handled for you.
/// 
/// ```rust
/// use eant2::*;
/// let train = EANT2::builder()
///   .inputs(10)
///   .outputs(3)
///   .build();
/// 
/// let (network, fitness) = train.run();
/// ```
/// 
/// # Advanced Example
/// 
/// ```rust
/// use eant2::*;
/// 
/// let eant = EANT2::builder()
///   .inputs(10)
///   .outputs(3)
///   .activation(Activation::Relu) // use the ReLu activation function
///   .print()                      // prints optimization progress.
///   .exploration(                 // EANT2 options (structural optimization)
///     Exploration::builder()
///       .population(20)           // 20 networks in the population
///       .offspring(4)             // each network spawns 4 offspring
///       .similar_fitness(0.2)     // network 'similarity' threshold
///       .terminate(
///          EANT2Termination::builder()
///            .fitness(0.15)       // either terminate EANT2 when fitness hits 0.15,
///            .generations(12)     // or terminate after 12 generations
///       )
///       .mutation_probabilities(
///         MutationProbabilities::zeros()
///           .add_connection(2.)    
///           .remove_connection(2.)
///           .add_neuron(1.)       
///           .add_bias(1.)         
///           .build().unwrap()
///       )
///       .build()
///    )
///    .exploitation(               // CMA-ES options (parameter optimization)
///      Exploitation::builder()
///        .restart(RestartStrategy::BIPOP(Default::default())) // custom restart strategy for CMA-ES
///        .terminate(
///          CMAESTermination::builder()
///            .fitness(0.15)   // terminate CMA-ES when fitness hits 0.15 (defaults to EANT2 fitness)
///            .generations(40) // force terminate CMA-ES after 40 CMA-ES generations (defaults to no limit)
///        )
///        .build()
///    )
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

  /// Activation function the network uses.
  #[builder(default = DEFAULT_ACTIVATION, setter(doc = "Activation function the network uses."))]
  pub activation: Activation,

  /// Initial network
  #[builder(default = None, setter(strip_option, doc = "Initial network."))]
  pub seed: Option<Network>,

  /// Structural exploration (EANT2: mutation) options
  #[builder(default_code = "Exploration::builder().build()", setter(doc = "Structural exploration (EANT2: mutation) options"))]
  pub exploration:  Exploration,

  /// Structural exploitation (CMA-ES: parameter tuning) options
  #[builder(default_code = "Exploitation::builder().build()", setter(doc = "Structural exploitation (CMA-ES: parameter tuning) options"))]
  pub exploitation: Exploitation,

  /// Print information throughout the optimization process.
  #[builder(setter(strip_bool, doc = "Print information throughout the optimization process."))]
  pub print: bool
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
      generation = select::select(self.exploration.population, &generation.individuals[..], self.exploration.similar_fitness);
      generation.individuals.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

      let best = &generation.individuals[0];
      
      // 3. Check EANT2 termination conditions
      if best.fitness <= self.exploration.terminate.fitness || g + 1 >= self.exploration.terminate.generations {
        if self.print {
          println!("EANT2 terminated in {} generations", g + 1);
          println!("Solution found with size {} and {} fitness", best.network.size + 1, best.fitness);
        }

        return (best.network.clone(), best.fitness);
      }

      if self.print { println!("Current best fitness: {}", best.fitness); }

      // 4. Add new individuals to the population by mutating the existing ones
      // TODO: consider parallelizing this step
      let mut new_individuals = Vec::with_capacity(self.exploration.offspring * self.exploration.population);
      for individual in generation.individuals.iter() {
        for _ in 0..self.exploration.offspring {
          let mut new = individual.clone();
          mutate(&mut new, &self.exploration.mutation_probabilities);
          
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

use super::{Exploitation, CMAESTermination};

  #[test]
 fn test_builder() {
    // let x = EANT2::builder()
    //   .inputs(10)
    //   .outputs(3)
    //   .build();
  }
}
