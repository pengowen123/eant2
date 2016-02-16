use std::cmp::Ordering;

use compare::*;
use utils::*;

// For preserving diversity
const MAX_DUPLICATES: usize = 1;
const MAX_SIMILAR: usize = 2;

pub fn select(population_size: usize, mut individuals: Vec<Individual>) -> Vec<Individual> {
    let mut pre_generation = Vec::new();

    for individual in &individuals {
        let mut new = individual.clone();

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

        new.duplicates -= 1;

        pre_generation.push(new);
    }

    let mut generation = Vec::new();
    let mut deleted = 0;

    for individual in &pre_generation {
        if !(individual.duplicates > MAX_DUPLICATES || individual.similar > MAX_SIMILAR) {
            generation.push(individual.clone());
        }

        if deleted >= pre_generation.len() - population_size {
            break;
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
