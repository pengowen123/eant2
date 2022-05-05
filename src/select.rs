use crate::generation::Generation;
use crate::utils::Individual;
use crate::FitnessFunction;
use cge::gene::Gene;
use cge::gene::GeneExtras::*;

// For preserving diversity
const MAX_DUPLICATES: usize = 1;
const MAX_SIMILAR: usize = 2;

/// 1. Greedily remove as many (purely structural) duplicates / too-similar networks as possible
/// 2. Take the remaining individuals who are most fit
// TODO: this is `O(n^2)`, optimization may be possible.
// TODO: I am thinking that if we can figure out a workable network "fingerprint", this could run in `O(n)`,
// TODO: and other speedups could be achieved.
// this is called serially in the main EANT2 algorithm,
// so (for now) there would be some benefit to parallelizing it. I got stuck doing this.
pub fn select<T: FitnessFunction + Clone + Send>(
    population_size: usize,
    individuals: &[Individual<T>],
    threshold: f64,
) -> Generation<T> {
    let mut pre_generation = Vec::with_capacity(individuals.len());

    for individual in individuals {
        let mut new = individual.clone();

        new.duplicates = 0;
        new.similar = 0;

        for individual_2 in individuals {
            match compare(&individual, &individual_2) {
                Category::Duplicate => new.duplicates += 1,
                Category::Similar => new.similar += 1,
                Category::Unique => {}
            }
        }

        // there is guaranteed to be one duplicate because it compares each individual to itself decrement to counter this
        new.duplicates -= 1;
        pre_generation.push(new);
    }

    let mut generation = Vec::new();
    let mut deleted = 0;

    // greedily skips individuals with too many duplicates, too many similar
    // until it cannot continue without making the population size drop below the configured size.
    for individual in &pre_generation {
        if !(individual.duplicates > MAX_DUPLICATES || individual.similar > MAX_SIMILAR)
            || deleted >= pre_generation.len() - population_size
        {
            generation.push(individual.clone());
        } else {
            deleted += 1;
        }
    }

    generation.sort_by(|a, b| {
        let by_fitness = a.fitness.partial_cmp(&b.fitness).unwrap();
        let similar_fitness = compare_fitness(&a, &b, threshold);

        if similar_fitness {
            a.network.size.partial_cmp(&b.network.size).unwrap()
        } else {
            by_fitness
        }
    });

    Generation {
        individuals: generation[0..population_size].to_vec(),
    }
}

pub enum Category {
    /// The network is a structural duplicate of another
    Duplicate,

    /// The network is structurally similar to another
    Similar,

    /// The network is structurally unique
    Unique,
}

pub fn compare<T: FitnessFunction + Clone>(a: &Individual<T>, b: &Individual<T>) -> Category {
    // this can be simplified a lot (and probably sped up a bit).
    // -> we say that two networks cannot be duplicates if they have differing gene "variants" in corresponding position.
    // -> I suspect a lot of this calculation can be done on a compressed representation, a 'fingerprint' of the network,
    //    much like a hash which can be easily updated upon mutation & checked must faster than this naive approach.
    // -> Maybe a fool's errand, you decide.

    // if we know it is a duplicate already, escape early
    if is_duplicate(&a.network.genome, &b.network.genome) {
        return Category::Duplicate;
    }

    // get an iterator over the neuron *pairs*, e.g. the `i`th neuron in network A and the `i`th neuron in network B.
    // written in this funky way to avoid two vector allocations that used to be used here.
    let is_neuron = |g: &&Gene| matches!(g.variant, Neuron(_, _));
    let neurons = a
        .network
        .genome
        .iter()
        .filter(is_neuron)
        .zip(b.network.genome.iter().filter(is_neuron));

    for (gene_a, gene_b) in neurons {
        // if the `i`th neuron-gene in A and B don't have the same ID, we immediately claim they are structurally unique.
        if gene_a.id != gene_b.id {
            return Category::Unique;
        }
    }

    // if neither duplicate nor unique, then they are similar.
    Category::Similar
}

/// Are the genomes `a` and `b` structurally identical?
pub fn is_duplicate(a: &[Gene], b: &[Gene]) -> bool {
    // Observation: if the networks are different lengths, this immediately implies that they cannot be structural duplicates.
    if a.len() != b.len() { return false; }

    for (gene_a, gene_b) in a.iter().zip(b.iter()) {
        if variant(gene_a) != variant(gene_b) {
            return false;
        }
    }

    false
}

pub fn compare_fitness<T>(a: &Individual<T>, b: &Individual<T>, threshold: f64) -> bool
where
    T: FitnessFunction + Clone,
{
    let diff = (a.fitness - b.fitness).abs();

    diff / b.fitness < threshold
}

fn variant(gene: &Gene) -> u8 {
    match gene.variant {
        Neuron(_, _) => 0,
        Input(_) => 1,
        Forward => 2,
        Recurrent => 3,
        Bias => 4,
    }
}
