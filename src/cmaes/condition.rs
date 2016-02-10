//! An enum to represent the conditions for terminating the CMA-ES algorithm.

pub enum CMAESEndConditions {
    // Maybe add a few more here, including Never
    /// Terminate if best fitness changes by less than some amount for some amount of generations.
    /// Usage: Stabilized(/* fitness */, /* generations */)
    Stabilized(f64, usize),

    /// Terminate if best fitness is under some amount.
    /// Usage: FitnessThreshold(/* fitness */)
    FitnessThreshold(f64),

    /// Terminate after the generation count reaches a number.
    /// Usage: MaxGenerations(/* generations */)
    MaxGenerations(usize),

    /// Terminate after calling the fitness function some amount of times.
    /// Usage: MaxEvaluations(/* calls */)
    MaxEvaluations(usize)
}
