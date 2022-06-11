use cge::gene::{Gene, InputId, NeuronId};

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::generation::Generation;
use crate::utils::{self, Individual};
use crate::{FitnessFunction, Network};

/// The maximum number of copies of each unique individual in the population allowed to survive
/// selection. When set to one, no duplicates are allowed (only one unique copy may survive). If
/// `force_meet_population_size` is set, this constraint may be violated if it would cause too few
/// individuals to survive to meet the target population size.
//
// The paper was not entirely clear about whether this should be set to one or two, but one seems
// more likely and was chosen, as retaining any duplicates probably provides little to no benefit on
// average (they effectively serve only as additional CMA-ES runs with potentially differing initial
// means).
const MAX_COPIES: usize = 1;
/// The maximum number of similar individuals allowed to survive for each unique individual in the
/// population. When set to two, for a given network, only one additional network similar to it
/// may survive, so if three or more networks are similar to each other, some will be culled. If
/// `force_meet_population_size` is set, this constraint may be violated if it would cause too few
/// individuals to survive to meet the target population size.
//
// The paper was not entirely clear about whether this should be set to two or three, but two is
// more consistent with the above interpretation of `MAX_COPIES` and was chosen.
const MAX_SIMILAR: usize = 2;

/// The relationship between networks in a `NetworkGroup`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupKind {
    /// The networks in the group are structurally similar to each other.
    Similar,
    /// The networks in the group are structurally identical to each other.
    Duplicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NetworkId(usize);

/// A group of networks that tracks how many networks have been taken from it so far.
#[derive(Debug)]
struct NetworkGroup {
    /// The relationship between networks in this group.
    kind: GroupKind,
    /// The IDs of the networks in this group.
    network_ids: Vec<NetworkId>,
    /// The number of networks taken from this group.
    taken: usize,
}

impl NetworkGroup {
    fn new(kind: GroupKind) -> Self {
        Self {
            kind,
            network_ids: Vec::new(),
            taken: 0,
        }
    }

    /// Adds the network ID to this `NetworkGroup`.
    fn push(&mut self, id: NetworkId) {
        self.network_ids.push(id)
    }

    /// Sorts this `NetworkGroup` by the criteria represented by `compare`.
    fn sort<T>(
        &mut self,
        network_map: &HashMap<NetworkId, Individual<T>>,
        similar_fitness_threshold: f64,
    ) where
        T: FitnessFunction + Clone,
    {
        self.network_ids.sort_by(|a, b| {
            compare(
                &network_map[&a],
                &network_map[&b],
                similar_fitness_threshold,
            )
        });
    }

    /// Returns the ID of the best network in this `NetworkGroup` that is valid to select if one
    /// exists. Networks for which the maximum number of similar networks or copies have already
    /// been selected will not be returned.
    fn best_constrained(
        &self,
        similar_groups: &[NetworkGroup],
        max_similar: usize,
        duplicate_groups: &[NetworkGroup],
        max_copies: usize,
    ) -> Option<NetworkId> {
        // Check that networks may still be removed from this group
        if self.kind == GroupKind::Similar && self.num_taken() >= max_similar {
            return None;
        } else if self.kind == GroupKind::Duplicate && self.num_taken() >= max_copies {
            return None;
        }

        self.network_ids
            .iter()
            .filter(|id| {
                // Check other groups containing the ID to see whether the maximum number of
                // networks have already been taken from them (which would mean the ID should not
                // be removed from this one, as it would also need to be removed from the others,
                // violating their constraints)
                let constrained_by_any_group = |groups: &[NetworkGroup], max_per_group| {
                    for g in groups {
                        // Check whether the group's constraint on the network (if any) has already
                        // been reached
                        if g.contains(**id) {
                            if g.num_taken() >= max_per_group {
                                return true;
                            }

                            // Networks may only be in one group per group category
                            break;
                        }
                    }

                    false
                };

                match self.kind {
                    // If this is a Similar group, only check Duplicate groups (this is the only
                    // group containing the ID in the Similar category)
                    GroupKind::Similar => !constrained_by_any_group(duplicate_groups, max_copies),
                    // If this is a Duplicate group, only check Similar groups (this is the only
                    // group containing the ID in the Duplicate category)
                    GroupKind::Duplicate => !constrained_by_any_group(similar_groups, max_similar),
                }
            })
            .next()
            .cloned()
    }

    /// Returns whether this `NetworkGroup` contains no networks.
    fn is_empty(&self) -> bool {
        self.network_ids.is_empty()
    }

    /// Returns the number of networks taken from this group so far.
    fn num_taken(&self) -> usize {
        self.taken
    }

    /// Returns whether this `NetworkGroup` contains the ID.
    fn contains(&self, id: NetworkId) -> bool {
        self.network_ids.contains(&id)
    }

    /// Removes the given ID from this `NetworkGroup` if it exists. Returns whether the ID was
    /// found.
    fn find_and_remove(&mut self, id: NetworkId) -> bool {
        if let Some((index, _)) = self.network_ids.iter().enumerate().find(|(_, x)| **x == id) {
            self.network_ids.remove(index);
            self.taken += 1;
            true
        } else {
            false
        }
    }
}

/// Applies the selection operator to the population. By default, no more than `MAX_COPIES`
/// instances of an individual and no more than `MAX_SIMILAR` structurally similar individuals will
/// be kept in the population per unique individual, where structurally similar means that two
/// networks share the same base neuron structure regardless of their connections.
///
/// The next generation will be no larger than `target_population_size`. However, it may be
/// smaller than this value if `individuals.len() < target_population_size`. It may also be smaller
/// due to the `MAX_COPIES` and `MAX_SIMILAR` constraints above. If this is not desired,
/// `force_meet_population_size` may be set to `true` to gradually relax these constraints
/// until the target size is met (or the entire population is selected).
///
/// Uses `similar_fitness_threshold` (an absolute difference) to determine whether two networks have
/// similar fitness values.
//
// NOTE: The algorithm for the selection operator implemented here is described below:
//
// 1. Sort all networks into groups of networks that are similar to and duplicates of each other.
//     a. For each network:
//         1. Compare it to the first network of each Duplicate group and add it to the group if the
//            two are duplicates (this step requires that structural identicality is
//            transitive).
//         2. If not added to an existing Duplicate group, create a new one and add the network to
//            it.
//         3. Compare it to the first network of each Similar group and add it to the group if the
//            two are either similar or duplicates (this step requires that this comparison as a
//            whole is transitive).
//         4. If not added to an existing Similar group, create a new one and add the network to it.
//     b. Sort each group according to the ranking criteria (prefer better fitness, or smaller size
//        if two fitness values are similar).
// 2. Until the target population size is met or all networks have been selected:
//     a. Find the highest-ranked network across all groups such that from no group containing the
//        network has the maximum number of networks for that group's type been selected so far:
//         1. From Similar groups, no more than `max_similar` networks may be selected
//         2. From Duplicate groups, no more than `max_copies` networks may be selected
//     b. If an individual was found, select it, increment the number of networks selected from each
//        group containing it, and remove it from those groups.
//     c. Otherwise, terminate (this will cause the target population size to not be reached).
//     d. Optionally, instead of terminating early, the constraints may be gradually relaxed until
//        the target population size is reached:
//         1. If there exists a non-empty Duplicate group from which fewer than `max_copies`
//            networks have been selected, increment `max_similar`.
//         2. Otherwise, increment `max_copies`.
//
// This algorithm should be O(N^2 * L) + O(N^3) worst case, where N is the length of `individuals`
// and L is the average size of the networks in the population. On average however, much of the work
// will be skipped or stopped early, and the only O(N^3) operations are `usize` comparisons, so the
// performance should still be good in general.
pub fn select<T: FitnessFunction + Clone + Send>(
    individuals: Vec<Individual<T>>,
    target_population_size: usize,
    similar_fitness_threshold: f64,
    force_meet_population_size: bool,
) -> Generation<T> {
    let mut individuals = individuals
        .into_iter()
        .enumerate()
        .map(|(i, x)| (NetworkId(i), x))
        .collect::<HashMap<_, _>>();

    // A list of groups of networks that are structurally identical to each other
    // In groups with only one individual, that individual has no duplicates
    // All networks are in this list, even if they are alone in their group
    let mut duplicate: Vec<NetworkGroup> = Vec::new();
    // A list of groups of networks that are structurally similar or identical to each other
    // Identical networks are considered similar as well in order to make similarity transitive
    // In groups with only one individual, that individual has no similar networks
    // All networks are in this list, even if they are alone in their group
    let mut similar: Vec<NetworkGroup> = Vec::new();

    // Place all networks into the correct groups in each category
    for (id, individual) in &individuals {
        let id = *id;

        // Check whether the network falls into any existing Duplicate group
        let mut added_to_existing_duplicate_group = false;
        for g in &mut duplicate {
            // Only the first network in each group needs to be checked because identicality is
            // transitive
            let existing_id = g.network_ids[0];
            let existing = &individuals[&existing_id];

            if let Similarity::Duplicate = check_similarity(&individual.network, &existing.network) {
                g.push(id);
                added_to_existing_duplicate_group = true;
                // Networks cannot be duplicates in more than one group because each group is
                // unique
                break;
            }
        }

        // Otherwise, place it in a new Duplicate group
        if !added_to_existing_duplicate_group {
            let mut new = NetworkGroup::new(GroupKind::Duplicate);
            new.push(id);
            duplicate.push(new);
        }

        // Check whether the network falls into any existing Similar group
        let mut added_to_existing_similar_group = false;
        for g in &mut similar {
            // Only the first network in each group needs to be checked because similarity or
            // identicality is transitive
            let existing_id = g.network_ids[0];
            let existing = &individuals[&existing_id];

            let category = check_similarity(&individual.network, &existing.network);

            if let Similarity::Similar | Similarity::Duplicate = category {
                g.push(id);
                added_to_existing_similar_group = true;
                // Networks cannot be similar in more than one group because each group is
                // unique
                break;
            }
        }

        // Otherwise, place it in a new Similar group
        if !added_to_existing_similar_group {
            let mut new = NetworkGroup::new(GroupKind::Similar);
            new.push(id);
            similar.push(new);
        }
    }

    // The total number of IDs across all groups should be double the number of individuals because
    // each individual is in exactly one group in each category (one in `similar`, one in
    // `duplicate`)
    let num_ids = similar.iter().chain(&duplicate).flat_map(|g| g.network_ids.iter()).count();
    assert_eq!(individuals.len() * 2, num_ids);

    // Sort groups internally by the criteria given by `compare`
    for g in &mut similar {
        g.sort(&individuals, similar_fitness_threshold);
    }

    for g in &mut duplicate {
        g.sort(&individuals, similar_fitness_threshold);
    }

    // Select networks for the next generation
    let mut selected_ids = Vec::with_capacity(target_population_size);
    let mut max_similar = MAX_SIMILAR;
    let mut max_copies = MAX_COPIES;

    // Loop until either the target population size is reached or the entire population has been
    // selected
    while selected_ids.len() < target_population_size && selected_ids.len() < individuals.len() {
        // The best individual that is valid to remove according to the similarity/uniqueness
        // constraints
        let best_constrained = similar
            .iter()
            .chain(&duplicate)
            .filter_map(|g| g.best_constrained(&similar, max_similar, &duplicate, max_copies))
            .min_by(|a, b| compare(&individuals[a], &individuals[b], similar_fitness_threshold));

        if let Some(id) = best_constrained {
            // If a best individual was found, select it

            // Remove selected IDs from all groups
            for g in similar.iter_mut().chain(&mut duplicate) {
                g.find_and_remove(id);
            }

            selected_ids.push(id);
        } else if force_meet_population_size {
            // Otherwise, relax constraints to meet the population size if the option is set

            // Check whether there exists a Duplicate group that is non-empty and has not reached
            // `max_copies` yet
            // If there is, the Duplicate group cannot be removed from not because `max_copies`
            // has been reached, but because a Similar group containing the network has reached
            // `max_similar`, which implies that increasing `max_similar` will allow a network to be
            // chosen
            let constrained_by_max_similar = duplicate
                .iter()
                .any(|g| !g.is_empty() && g.num_taken() < max_copies);

            if constrained_by_max_similar {
                max_similar += 1;
            } else {
                // Otherwise, all non-empty Duplicate groups have reached `max_copies`, so it must
                // be increased
                max_copies += 1;
            }
        } else {
            // Otherwise, if the option is not set, end the selection process early
            break;
        }
    }

    // If `force_meet_population_size` is set, the target population size should be met regardless
    // of similarity/uniqueness constraints, unless there are too few individuals in the population
    // to meet it, in which case all individuals are selected
    assert!(
        !force_meet_population_size
            || selected_ids.len() == individuals.len()
            || selected_ids.len() == target_population_size
    );

    // Return the selected networks
    let selected = selected_ids
        .into_iter()
        .map(|id| individuals.remove(&id).unwrap())
        .collect();

    Generation {
        individuals: selected,
    }
}

/// The similarity relationship between two networks.
#[derive(Debug)]
enum Similarity {
    /// The two networks are structurally identical.
    Duplicate,
    /// The two networks are structurally similar (only neuron structure is considered; connections
    /// are ignored).
    Similar,
    /// The two networks are neither structurally identical nor similar.
    Unique,
}

/// Compares the two networks and returns their relationship to each other.
fn check_similarity(a: &Network, b: &Network) -> Similarity {
    let is_similar = is_similar(a, b);
    // Similarity is a requirement for `is_duplicate`
    let is_duplicate = is_similar && is_duplicate(a, b);

    if is_duplicate {
        Similarity::Duplicate
    } else if is_similar {
        Similarity::Similar
    } else {
        Similarity::Unique
    }
}

/// Returns whether the two networks are structurally similar, where similar means
/// that the neuron structure of each gene is identical, ignoring all connections. Attempts to be as
/// general as feasible with regards to checking graph isomorphism instead of genome identicality,
/// but compromises significantly for practicality.
fn is_similar(a: &Network, b: &Network) -> bool {
    let get_neuron_structure = |network: &Network| {
        let mut implicit_connections = HashSet::new();

        for (gene, parent) in network.genome().iter().zip(network.parents()) {
            if let Some(neuron) = gene.as_neuron() {
                if let Some(parent_id) = parent {
                    // Insert a connection between each neuron and its parent
                    implicit_connections.insert((*parent_id, neuron.id()));
                }
            }
        }

        implicit_connections
    };

    get_neuron_structure(a) == get_neuron_structure(b)
}

/// Returns whether the two networks are structurally identical. Requires that
/// `is_similar(a, b) == true`. Attempts to be as general as feasible with regards to checking graph
/// isomorphism instead of genome identicality, but compromises significantly for practicality.
fn is_duplicate(a: &Network, b: &Network) -> bool {
    // Observation: if the networks are different lengths, this immediately implies that they cannot
    // be structural duplicates.
    if a.len() != b.len() {
        return false;
    }

    // For each neuron ID, check that the corresponding neuron in each network contains the same
    // incoming connections (the networks are assumed to be structurally similar, so they also
    // contain the same IDs)
    for id in a.neuron_ids() {
        let get_children_unordered = |network, id| {
            let mut has_bias = false;
            let mut inputs: HashSet<InputId> = HashSet::new();
            let mut forward_jumpers: HashSet<NeuronId> = HashSet::new();
            let mut recurrent_jumpers: HashSet<NeuronId> = HashSet::new();

            for child in utils::get_direct_children(network, id) {
                match child {
                    Gene::Bias(_) => has_bias = true,
                    Gene::Input(input) => {
                        inputs.insert(input.id());
                    }
                    Gene::ForwardJumper(forward) => {
                        forward_jumpers.insert(forward.source_id());
                    }
                    Gene::RecurrentJumper(recurrent) => {
                        recurrent_jumpers.insert(recurrent.source_id());
                    }
                    Gene::Neuron(_) => {}
                }
            }

            (has_bias, inputs, forward_jumpers, recurrent_jumpers)
        };

        if get_children_unordered(a, id) != get_children_unordered(b, id) {
            return false;
        }
    }

    true
}

/// Compares the two individuals for sorting. The individual with better fitness is ranked higher,
/// unless the two individuals have fitness values within `similar_fitness_threshold` of each other,
/// in which case the smaller individual is ranked higher instead.
fn compare<T>(a: &Individual<T>, b: &Individual<T>, similar_fitness_threshold: f64) -> Ordering
where
    T: FitnessFunction + Clone,
{
    if similar_fitness(a, b, similar_fitness_threshold) {
        a.network.len().cmp(&b.network.len())
    } else {
        a.fitness.partial_cmp(&b.fitness).unwrap()
    }
}

/// Returns whether the absolute difference between the two individuals' fitness values is below
/// `threshold`.
fn similar_fitness<T>(a: &Individual<T>, b: &Individual<T>, threshold: f64) -> bool
where
    T: FitnessFunction + Clone,
{
    if a.fitness.is_none() || b.fitness.is_none() {
        return false;
    }

    let fitness_a = a.fitness.unwrap();
    let fitness_b = b.fitness.unwrap();

    (fitness_a - fitness_b).abs() < threshold
}
