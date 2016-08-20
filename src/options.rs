//! Option types for the EANT2 algorithm.

use cmaes::options::CMAESEndConditions;
use cge::{Network, TransferFunction};

const DEFAULT_CMAES_CONDITIONS: [CMAESEndConditions; 2] = [
    CMAESEndConditions::StableGenerations(0.0001, 5),
    CMAESEndConditions::MaxGenerations(500)
];

pub const DEFAULT_POPULATION_SIZE: usize = 10;
pub const DEFAULT_OFFSPRING_COUNT: usize = 4;
pub const DEFAULT_SIMILAR_FITNESS: f64 = 0.15;
pub const DEFAULT_MIN_FITNESS: f64 = 0.0;
pub const DEFAULT_MAX_GENERATIONS: usize = 30;
pub const DEFAULT_THREADS: usize = 0;
pub const DEFAULT_CMAES_RUNS: usize = 2;
pub const DEFAULT_PRINT_OPTION: bool = false;
pub const DEFAULT_WEIGHTS: [usize; 4] = [3, 8, 1, 3];
pub const DEFAULT_SEED: Option<Network> = None;
pub const DEFAULT_TRANSFER_FUNCTION: TransferFunction = TransferFunction::Sigmoid;

/// A container for all parameters and options for the EANT2 algorithm. See constants for default
/// values.
///
/// Using default options is convenient, but may slightly reduce the quality of the neural networks
/// generated, and can increase the time needed to find a solution. It is recommended to adjust the
/// parameters depending on whether it is more important to find a solution quickly or to find a
/// better solution. Note that increasing population size and CMA-ES runs will only affect the
/// quality a small amount, so using smaller values will greatly speed up the algorithm, and does
/// not have a large downside. It is important to set the fitness threshold using the
/// `fitness_threshold` method, otherwise the algorithm may terminate too soon or take a long time to
/// run. Here are some general tips for setting the options (more in the documentation):
/// 
/// If a minimal neural network is preferred, increase `similar_fitness` and increase the weight for
/// connection removal. For complex problems, decrease population size and the weights for adding
/// connections and neurons.
///
/// # Examples
///
/// ```
/// use eant2::*;
///
/// // A set of default options, with 2 inputs and 3 outputs to each neural network
/// let options = EANT2Options::new(2, 3);
///
/// // A set of options with 4 CMA-ES optimizations per individual, a minimum fitness of 10.0, and
/// // a population size of 50
/// let options = EANT2Options::new(2, 3)
///     .fitness_threshold(10.0)
///     .fitness_threshold(0.0)
///     .cmaes_runs(4)
///     .population_size(50);
/// ```
#[derive(Clone)]
pub struct EANT2Options {
    pub inputs: usize,
    pub outputs: usize,
    pub population_size: usize,
    pub offspring_count: usize,
    pub fitness_threshold: f64,
    pub similar_fitness: f64,
    pub max_generations: usize,
    pub threads: usize,
    pub cmaes_runs: usize,
    pub cmaes_end_conditions: Vec<CMAESEndConditions>,
    pub print_option: bool,
    pub weights: [usize; 4],
    pub seed: Option<Network>,
    pub transfer_function: TransferFunction
}

impl EANT2Options {
    /// Returns a set of default options (see constants for default values).
    pub fn new(inputs: usize, outputs: usize) -> EANT2Options {
        if inputs == 0 || outputs == 0 {
            panic!("Neural network inputs and outputs cannot be zero");
        }

        EANT2Options {
            inputs: inputs,
            outputs: outputs,
            population_size: DEFAULT_POPULATION_SIZE,
            offspring_count: DEFAULT_OFFSPRING_COUNT,
            fitness_threshold: DEFAULT_MIN_FITNESS,
            similar_fitness: DEFAULT_SIMILAR_FITNESS,
            max_generations: DEFAULT_MAX_GENERATIONS,
            threads: DEFAULT_THREADS,
            cmaes_runs: DEFAULT_CMAES_RUNS,
            cmaes_end_conditions: DEFAULT_CMAES_CONDITIONS.to_vec(),
            print_option: DEFAULT_PRINT_OPTION,
            weights: DEFAULT_WEIGHTS,
            seed: DEFAULT_SEED,
            transfer_function: DEFAULT_TRANSFER_FUNCTION
        }
    }

    /// Add docs here
    pub fn seed(mut self, network: Network) -> EANT2Options {
        self.seed = Some(network);
        self
    }
    
    /// Sets the population size. Increasing this option may produce higher quality neural
    /// networks, but will increase the time needed to find a solution.
    pub fn population_size(mut self, size: usize) -> EANT2Options {
        if size == 0 {
            panic!("Population size cannot be zero");
        }

        self.population_size = size;
        self
    }

    /// Sets the offspring count (how many offspring each individual spawns). Increasing this
    /// option may produce higher quality neural networks, but will increase the time needed to
    /// find a solution.
    pub fn offspring_count(mut self, count: usize) -> EANT2Options {
        self.offspring_count = count;
        self
    }

    /// Sets the fitness threshold. The algorithm terminates if the best individual has a fitness
    /// less than or equal to the specified amount. It is recommended to set this option to a value
    /// specific to each fitness function. The specified fitness may not be reached if the max
    /// generation is reached first.
    pub fn fitness_threshold(mut self, fitness: f64) -> EANT2Options {
        self.fitness_threshold = fitness;
        self
    }

    /// Sets the maximum generations. The algorithm terminates after the specified number of
    /// generations pass. Increasing this option will likely produce higher quality networks, but
    /// will increase the running time of the algorithm. The number of generations specified may
    /// not be reached if the fitness threshold is reached first.
    pub fn max_generations(mut self, generations: usize) -> EANT2Options {
        self.max_generations = generations;
        self
    }

    /// Sets the threshold for deciding whether two neural networks have a similar fitness.
    /// Increasing this option will make the algorithm more aggresively prefer smaller
    /// neural networks. Decreasing it will do the opposite, allowing larger individuals to stay in
    /// the population. It is recommended to set this option higher if a small neural network is
    /// preferred. The downside is it will take slightly longer to find a solution, due to more
    /// higher fitness neural networks being discarded.
    pub fn similar_fitness(mut self, threshold: f64) -> EANT2Options {
        self.similar_fitness = threshold;
        self
    }

    /// Sets the number of threads to use in the algorithm. Increasing this option on multi-core
    /// hardware will greatly decrease the running time of the algorithm.
    pub fn threads(mut self, threads: usize) -> EANT2Options {
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
    /// the documentation for it is available [here][1].
    ///
    /// [1]: http://pengowen123.github.io/cmaes/cmaes/options/enum.CMAESEndConditions.html
    ///
    /// # Examples
    ///
    /// ```
    /// use eant2::*;
    ///
    /// let cmaes_conditions = vec![CMAESEndConditions::MaxGenerations(250),
    ///                             CMAESEndConditions::StableGenerations(0.001, 5)];
    /// 
    /// let options = EANT2Options::new(2, 2)
    ///                   .cmaes_end_conditions(cmaes_conditions);
    pub fn cmaes_end_conditions(mut self, conditions: Vec<CMAESEndConditions>) -> EANT2Options {
        self.cmaes_end_conditions = conditions;
        self
    }

    /// Sets whether to print info while the algorithm is running. The generation number is
    /// printed, and info about the solution is printed when the algorithm terminates.
    pub fn print(mut self, print: bool) -> EANT2Options {
        self.print_option = print;
        self
    }

    /// Sets the weight for choosing each mutation. The first element should be the weight for
    /// adding a connection, the second for removing a connection, the third for adding a
    /// neuron, and the fourth for adding a bias input. The weights are relative, so with the
    /// weights `[1, 2, 4, 8]`, each mutation has twice the chance of the previous. In general,
    /// the chance for connection mutations should be higher than the chance for a neuron mutation.
    /// If a minimal network is desired, set the bias, neuron and connection addition weights low,
    /// and the connection removal rate high. For complex problems where network size isn't an
    /// issue, high connection addition rate is a good idea.
    pub fn mutation_weights(mut self, weights: [usize; 4]) -> EANT2Options {
        self.weights = weights;
        self
    }

    /// Sets the transfer function to use for the trained neural networks. See the documentation
    /// [here][1] for information.
    ///
    /// [1]: http://pengowen123.github.io/cge/cge/transfer/enum.TransferFunction.html
    pub fn transfer_function(mut self, function: TransferFunction) -> EANT2Options {
        self.transfer_function = function;
        self
    }
}
