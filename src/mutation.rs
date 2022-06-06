use crate::mutation_probabilities::MutationSampler;
use crate::utils::Individual;
use crate::FitnessFunction;
use rand::{thread_rng, Rng};

/// The type of mutation to perform.
#[derive(Copy, Clone)]
pub enum MutationType {
    AddConnection,
    RemoveConnection,
    AddNode,
    AddBias,
}

/// Selects a random valid mutation and applies it to a network
pub fn mutate<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    probabilities: &MutationSampler,
) {
    let mut rng = thread_rng();

    match probabilities.sample(&mut rng) {
        MutationType::AddConnection => {}
        MutationType::AddNode => {}
        MutationType::AddBias => {}
        MutationType::RemoveConnection => {}
    }
}
