//! Option types for the EANT2 algorithm.

use cmaes::options::CMAESEndConditions;

const DEFAULT_CMAES_CONDITIONS: [CMAESEndConditions; 2] = [
    CMAESEndConditions::StableGenerations(0.0001, 5),
    CMAESEndConditions::MaxGenerations(500)
];

const DEFAULT_POPULATION_SIZE: usize = 30;
const DEFAULT_OFFSPRING_COUNT: usize = 2;
const DEFAULT_MAX_GENERATIONS: usize = 100;
const DEFAULT_CMAES_RUNS: usize = 2;

/// A container for all parameters and options for the EANT2 algorithm. Using default options is
/// convenient, but may slightly reduce the quality of the neural networks generated, and can
/// increase the time needed to find a solution. It is recommended to adjust the parameters
/// depending on whether it is more important to find a solution quickly or to find a better
/// solution. Note that increasing population size and CMA-ES runs will only affect the quality a
/// small amount, so using smaller values will greatly speed up the algorithm, and does not have a
/// large downside.
///
/// # Examples
///
/// ```
/// use eant2::*;
///
/// // A set of default options, with 2 inputs and 3 outputs to each neural network, and a minimum
/// // fitness of 10.0
/// let options = EANT2Options::new(2, 3, 10.0);
///
/// // A set of options with 2 CMA-ES optimizations per individual, and a population size of 50
/// let options = EANT2Options::new(2, 3, 10.0)
///     .cmaes_runs(2)
///     .population_size(50);
/// ```
#[derive(Clone)]
pub struct EANT2Options {
    pub inputs: usize,
    pub outputs: usize,
    pub population_size: usize,
    pub offspring_count: usize,
    pub min_fitness: f64,
    pub max_generations: usize,
    pub threads: u8,
    pub cmaes_runs: usize,
    pub cmaes_end_conditions: Vec<CMAESEndConditions>
}

impl EANT2Options {
    /// Returns a set of default options.
    pub fn new(inputs: usize, outputs: usize, min_fitness: f64) -> EANT2Options {
        if inputs == 0 || outputs == 0 {
            panic!("Neural network inputs and outputs cannot be zero");
        }

        EANT2Options {
            inputs: inputs,
            outputs: outputs,
            population_size: DEFAULT_POPULATION_SIZE,
            offspring_count: DEFAULT_OFFSPRING_COUNT,
            min_fitness: min_fitness,
            max_generations: DEFAULT_MAX_GENERATIONS,
            threads: 1,
            cmaes_runs: DEFAULT_CMAES_RUNS,
            cmaes_end_conditions: DEFAULT_CMAES_CONDITIONS.to_vec()
        }
    }
    
    /// Sets the population size
    pub fn population_size(mut self, size: usize) -> EANT2Options {
        if size == 0 {
            panic!("Population size cannot be zero");
        }

        self.population_size = size;
        self
    }

    /// Sets the offspring count (how many offspring each individual spawns)
    pub fn offspring_count(mut self, count: usize) -> EANT2Options {
        if count == 0 {
            panic!("Offspring count cannot be zero");
        }

        self.offspring_count = count;
        self
    }

    /// Sets the minimum fitness. The algorithm terminates if the best individual has a fitness of at
    /// least the specified amount.
    pub fn min_fitness(mut self, fitness: f64) -> EANT2Options {
        self.min_fitness = fitness;
        self
    }

    /// Sets the maximum generations. The algorithm terminates after the specified number of
    /// generations pass.
    pub fn max_generations(mut self, generations: usize) -> EANT2Options {
        self.max_generations = generations;
        self
    }

    /// Sets the number of threads to use in the algorithm.
    pub fn threads(mut self, threads: u8) -> EANT2Options {
        if threads == 0 {
            panic!("Threads cannot be zero");
        }

        self.threads = threads;
        self
    }

    /// Sets the number of times CMA-ES should be run on each individual. More runs will produce
    /// higher quality neural networks faster at the cost of a higher number of fitness function calls.
    /// Increase the number of runs if the fitness function is cheap to run, or if time taken to
    /// find a solution is not important.
    pub fn cmaes_runs(mut self, runs: usize) -> EANT2Options {
        if runs == 0 {
            panic!("CMA-ES runs cannot be zero");
        }

        self.cmaes_runs = runs;
        self
    }

    /// Sets the end conditions of CMA-ES. Setting higher stable generations or max generations or
    /// evaluations can produce higher quality neural networks at the cost of a longer time taken
    /// to find a solution. Decreasing these options is a good idea when it is important to find a
    /// solution quickly. DO NOT set the fitness threshold option, or it will take a very long time
    /// to find a solution, even on the simplest problems! It is recommended to leave this option alone, but
    /// the documentation for it is available [here](http://pengowen123.github.io/cmaes/cmaes/options/enum.CMAESEndConditions.html).
    pub fn cmaes_end_conditions(mut self, conditions: Vec<CMAESEndConditions>) -> EANT2Options {
        self.cmaes_end_conditions = conditions;
        self
    }
}
