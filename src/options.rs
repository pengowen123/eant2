use cmaes::options::CMAESEndConditions;

const DEFAULT_CMAES_CONDITION: CMAESEndConditions = CMAESEndConditions::StableGenerations(0.0001, 5);

/// Option type for the EANT2 algorithm.
///
/// # Examples
///
/// ```
/// // A set of default options, with 2 inputs and 3 outputs to each neural network
/// let options = EANT2Options::default(2, 3);
///
/// // A set of options with 2 CMA-ES optimizations per individual, and a minimum fitness to end
/// // after reaching of 10.0
/// let options = EANT2Options::custom(2, 3)
///     .cmaes_runs(2)
///     .min_fitness(2.0);
/// ```
#[derive(Clone)]
pub struct EANT2Options {
    pub inputs: usize,
    pub outputs: usize,
    pub population_size: usize,
    pub offspring_count: usize,
    pub min_fitness: f64,
    pub threads: u8,
    pub cmaes_runs: usize,
    pub cmaes_end_condition: CMAESEndConditions
}

impl EANT2Options {
    fn new(inputs: usize, outputs: usize, min_fitness: f64) -> EANT2Options {
        if outputs == 0 {
            panic!("Neural network outputs cannot be zero");
        }

        EANT2Options {
            inputs: inputs,
            outputs: outputs,
            population_size: 10,
            offspring_count: 2,
            min_fitness: min_fitness,
            threads: 1,
            cmaes_runs: 2,
            cmaes_end_condition: DEFAULT_CMAES_CONDITION,
        }
    }
}
