use std::cmp::Ordering;

use compare::*;
use utils::*;
use fitness::NNFitnessFunction;

// For preserving diversity
const MAX_DUPLICATES: usize = 1;
const MAX_SIMILAR: usize = 2;

pub fn select<T>(population_size: usize, individuals: Vec<Individual<T>>) -> Vec<Individual<T>>
    where T: NNFitnessFunction + Clone
{
    let mut pre_generation = Vec::new();

    for individual in &individuals {
        let mut new = individual.clone();

        new.duplicates = 0;
        new.similar = 0;

        for individual_2 in &individuals {
            let category = compare(&individual, &individual_2);

            match category {
                Category::Duplicate => {
                    new.duplicates += 1;
                },
                Category::Similar => {
                    new.similar += 1;
                },
                Category::Okay => {}
            }
        }

        // There is guaranteed to be one duplicate because it compares each individual to itself
        // Decrement it to counter this
        new.duplicates -= 1;

        pre_generation.push(new);
    }

    let mut generation = Vec::new();
    let mut deleted = 0;

    for individual in &pre_generation {
        if !(individual.duplicates > MAX_DUPLICATES || individual.similar > MAX_SIMILAR) ||
            deleted >= pre_generation.len() - population_size {
            generation.push(individual.clone());
        } else {
            deleted += 1;
        }
    }

    generation.sort_by(|a, b| {
        let by_fitness = match a.fitness.partial_cmp(&b.fitness).unwrap() {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => Ordering::Equal
        };

        let similar_fitness = compare_fitness(&a, &b);

        if similar_fitness {
            if a.network.size < b.network.size {
                Ordering::Greater
            } else if b.network.size < a.network.size {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        } else {
            by_fitness
        }
    });

    generation[0..population_size].to_vec()
}
