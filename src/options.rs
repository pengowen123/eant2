use cge::Activation;
use cmaes::{restart::{RestartStrategy}};
use typed_builder::TypedBuilder;
use crate::mutation_probabilities::MutationSampler;

pub(crate) const DEFAULT_ACTIVATION:          Activation       = Activation::Sigmoid;
pub(crate) const DEFAULT_POPULATION_SIZE:     usize            = 10;
pub(crate) const DEFAULT_OFFSPRING_COUNT:     usize            = 4;
pub(crate) const DEFAULT_SIMILAR_FITNESS:     f64              = 0.15;
pub(crate) const DEFAULT_MAX_GENERATIONS:     usize            = 30;
pub(crate) const DEFAULT_TERMINATING_FITNESS: f64              = 0.0;
pub(crate) const DEFAULT_EANT2_TERMINATION:   EANT2Termination = EANT2Termination { fitness: 0.0, generations: 30 };

/// When should the (outer) EANT2 algorithm terminate?
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

/// Exploration options. 
/// These are the options that control the structural exploration (EANT2: mutation).
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

/// When should CMA-ES (inner loop) terminate?
/// You usually don't need to configure this.
#[derive(TypedBuilder)]
pub struct CMAESTermination {
  #[builder(default = None, setter(strip_option, doc = "Default is to use the fitness from `EANT2Termination` options, which defaults to `0.0`."))]
  pub fitness:     Option<f64>,

  #[builder(default = None, setter(strip_option, doc = "Default is no CMA-ES generation limit."))]
  pub generations: Option<usize>
}

/// Exploitation options. 
/// These are the options that control parameter optimization (CMA-ES).
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