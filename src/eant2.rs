use cge::Activation;
use std::sync::Arc;
use typed_builder::TypedBuilder;

use crate::cge_utils::Network;
use crate::options::*;
use crate::{generation::Generation, mutation::mutate, select, FitnessFunction};

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
///
/// ```rust
/// use eant2::*;
/// let train = EANT2::builder()
///   .inputs(10)
///   .outputs(3)
///   .build();
///
/// let (network, fitness) = train.run(&MyFitnessFunction);
/// ```
///
/// # Advanced Example
///
/// - Most options have good default values, and exist only for flexibility.
///
/// ```rust
/// use eant2::*;
///
/// let eant = EANT2::builder()
///   .inputs(10)
///   .outputs(3)
///   .activation(Activation::Relu) // use the ReLu activation function
///   .print()                      // prints optimization progress
///   .exploration(                 // EANT2 options (structural optimization)
///     Exploration::builder()
///       .population(20)           // 20 networks in the population
///       .offspring(4)             // each network spawns 4 offspring
///       .similarity(0.2)          // network 'similarity' threshold
///       .terminate(
///         EANT2Termination::builder()
///           .fitness(0.15)        // either terminate EANT2 when best fitness hits 0.15,
///           .generations(12)      // or terminate after 12 generations
///       )
///       .mutation_probabilities(  // describe relative mutation probabilities
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
///            .evaluations(60) // force terminate CMA-ES if fitness function is evaluated 60 times (default no limit)
///            .generations(40) // force terminate CMA-ES after 40 CMA-ES generations (default no limit)
///        )
///        .build()
///    )
///   .build();
///
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
    #[builder(
        default_code = "Exploration::builder().build()",
        setter(doc = "Structural exploration (EANT2: mutation) options")
    )]
    pub exploration: Exploration,

    /// Structural exploitation (CMA-ES: parameter tuning) options
    #[builder(
        default_code = "Exploitation::builder().build()",
        setter(doc = "Structural exploitation (CMA-ES: parameter tuning) options")
    )]
    pub exploitation: Exploitation,

    /// Print information throughout the optimization process.
    #[builder(setter(
        strip_bool,
        doc = "Print information throughout the optimization process."
    ))]
    pub print: bool,
}

impl EANT2 {
    /// Run the optimization algorithm until termination conditions are met, yielding the best network and its fitness.
    pub fn run<T>(&self, object: &T) -> (Network, f64)
    where
        T: 'static + FitnessFunction + Clone + Send + Sync,
    {
        let object = Arc::new(object.clone());
        let mut g = 0;
        // Initialize a set of minimal networks
        let mut generation = Generation::initialize(&self, object);

        loop {
            if self.print {
                println!("Beginning EANT2 generation {}", g + 1);
            }

            // 1. Add new individuals to the population by mutating the existing ones
            // TODO: consider parallelizing this step
            let mut new_individuals =
                Vec::with_capacity((self.exploration.offspring + 1) * self.exploration.population);
            for mut individual in generation.individuals {
                // Increment gene ages
                for age in &mut individual.ages {
                    *age += 1;
                }

                // Carry over each individual to the next generation unchanged
                new_individuals.push(individual.clone());

                // Also mutate it to produce offspring
                for _ in 0..self.exploration.offspring {
                    let mut offspring = individual.clone();
                    mutate(&mut offspring, &self.exploration.mutation_probabilities);

                    new_individuals.push(offspring);
                }
            }
            generation.individuals = new_individuals;

            // 2. Get fitness of network topologies by optimizing the parameters of each individual
            // with CMA-ES to get their maximum potential.
            //    - This stage makes up for nearly all the running time of the algorithm, sometimes
            //    taking hours or days.
            generation.update_generation(&self);

            // 3. Select individuals to go on to the next generation
            generation = select::select(
                self.exploration.population,
                &generation.individuals[..],
                self.exploration.similarity,
            );

            let best = generation
                .individuals
                .iter()
                .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .unwrap();
            let best_fitness = best.fitness.unwrap();

            // 4. Check EANT2 termination conditions
            if best_fitness <= self.exploration.terminate.fitness
                || g + 1 >= self.exploration.terminate.generations
            {
                if self.print {
                    println!("EANT2 terminated in {} generations", g + 1);
                    println!(
                        "Solution found with size {} and {} fitness",
                        best.network.len(),
                        best_fitness,
                    );
                }

                return (best.network.clone(), best_fitness);
            }

            if self.print {
                println!("Current best fitness: {}", best.fitness.unwrap());
            }

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
