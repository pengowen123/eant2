//! EANT2's neural network mutation operators.

// The original paper was not entirely clear about all aspects of these operators, so a (hopefully)
// reasonable choice based on all the resources available on the algorithm was made in each case
// where there was ambiguity.
//
// The overall design of these mutation operators is as follows:
//
// - Step 1: Choose a mutation type
// - Step 2: Choose a mutation subtype if necessary
// - Step 3: Assemble a list of all valid mutations of the chosen type and subtype
// - Step 4: If the list is not empty, choose one and apply it
//
// This may result in "dead ends" being reached, where a type and subtype are chosen that have no
// valid mutations, while another type and/or subtype does have valid mutations. In theory, a 100%
// success rate could be achieved by calculating all valid mutations for each type before making any
// choices. However, this would add considerable complexity and could hurt performance, so a
// compromise was chosen instead (see the steps above). One exception to this rule is the
// `add_subnetwork` mutation, which simply chooses a random parent and tries to add a subnetwork to
// it, aborting if none would be valid. If all possibilities were enumerated according to step 3,
// aborting could be avoided in some cases, but the simpler solution was chosen here due to the
// extreme unlikeliness of no subnetworks being valid for a given parent.
//
// The time complexity of these mutations should generally be O(n^2) on the number of neurons in
// the network due to the calculation of valid mutations. However, networks should generally
// not be large enough for this to matter, as it quickly becomes infeasible to optimize larger
// networks with CMA-ES anyways. Additionally, basic tests reveal that these algorithms perform
// reasonably well on networks with up to several thousand genes, which is well beyond what CMA-ES
// can be expected to optimize. Regardless, the time taken in CMA-ES optimization and the fitness
// function will almost always dwarf any time spent mutating networks in practice.

use cge::gene::{
    Bias, ForwardJumper, Gene, Input, InputId, NeuronId, NonNeuronGene, RecurrentJumper,
};
use rand::rngs::ThreadRng;
use rand::seq::IteratorRandom;
use rand::{thread_rng, Rng};

use std::collections::HashSet;
use std::iter;

use crate::cge_utils::{Network, INITIAL_WEIGHT_VALUE};
use crate::mutation_probabilities::MutationSampler;
use crate::utils::Individual;
use crate::FitnessFunction;

/// The chance for each valid incoming or outgoing forward connection to/from a new subnetwork to be
/// added.
// TODO: Make this customizable
const NEW_SUBNETWORK_FORWARD_CONNECTION_PROBABILITY: f64 = 0.2;

/// The type of mutation to perform.
#[derive(Copy, Clone)]
pub enum MutationType {
    AddConnection,
    RemoveConnection,
    AddNode,
    AddBias,
}

/// Tries to apply a random mutation operator to the network. Returns whether any mutation was
/// actually performed.
pub fn mutate<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    probabilities: &MutationSampler,
) -> bool {
    let mut rng = thread_rng();

    match probabilities.sample(&mut rng) {
        MutationType::AddConnection => add_connection(individual, &mut rng),
        MutationType::AddNode => add_subnetwork(individual, &mut rng),
        MutationType::AddBias => add_bias(individual, &mut rng),
        MutationType::RemoveConnection => remove_connection(individual, &mut rng),
    }
}

/// Randomly adds a connection between two neurons or a neuron and a network input. Returns whether
/// any mutation was performed.
///
/// The original paper was not clear about whether input genes count as connections, but this
/// function assumes they do and therefore may add them.
fn add_connection<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    rng: &mut ThreadRng,
) -> bool {
    // TODO: Add option to customize the probabilities for these connection types
    match rng.gen_range(0..=2) {
        0 => add_forward_jumper(individual, rng),
        1 => add_recurrent_jumper(individual, rng),
        2 => add_input(individual, rng),
        _ => unreachable!(),
    }
}

/// Randomly adds a forward jumper gene to the network. Returns whether any mutation was performed.
fn add_forward_jumper<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    rng: &mut ThreadRng,
) -> bool {
    let network = &mut individual.network;

    // Find all valid, non-redundant forward jumper connections between neurons
    let valid_connections = network.neuron_ids().flat_map(|parent_id| {
        // The neuron IDs with greater depth than this neuron that aren't connected to it
        // explicitly or implicitly
        let parent_depth = network[parent_id].depth();
        let mut missing_connections = network
            .get_valid_forward_jumper_sources(parent_depth)
            .collect::<HashSet<_>>();

        for g in get_direct_children(network, parent_id) {
            if let Gene::ForwardJumper(forward) = g {
                // Check for explicit connections
                missing_connections.remove(&forward.source_id());
            } else if let Gene::Neuron(neuron) = g {
                // Check for implicit connections as well
                missing_connections.remove(&neuron.id());
            }
        }

        missing_connections
            .into_iter()
            .map(move |source_id| (parent_id, source_id))
    });

    // Choose one at random and add it
    if let Some((parent, source)) = valid_connections.choose(rng) {
        let input = ForwardJumper::new(source, INITIAL_WEIGHT_VALUE);
        add_non_neuron(individual, parent, input);
        true
    } else {
        false
    }
}

/// Randomly adds a recurrent jumper gene to the network. Returns whether any mutation was
/// performed.
fn add_recurrent_jumper<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    rng: &mut ThreadRng,
) -> bool {
    let network = &mut individual.network;

    // Find all non-redundant recurrent jumper connections between neurons
    let valid_connections = network.neuron_ids().flat_map(|parent_id| {
        // The neuron IDs that aren't connected to this neuron
        let mut missing_connections = network.neuron_ids().collect::<HashSet<_>>();

        for g in get_direct_children(network, parent_id) {
            if let Gene::RecurrentJumper(recurrent) = g {
                missing_connections.remove(&recurrent.source_id());
            }
        }

        missing_connections
            .into_iter()
            .map(move |source_id| (parent_id, source_id))
    });

    // Choose one at random and add it
    if let Some((parent, source)) = valid_connections.choose(rng) {
        let input = RecurrentJumper::new(source, INITIAL_WEIGHT_VALUE);
        add_non_neuron(individual, parent, input);
        true
    } else {
        false
    }
}

/// Randomly adds an input gene to the network. Returns whether any mutation was performed.
fn add_input<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    rng: &mut ThreadRng,
) -> bool {
    let network = &mut individual.network;

    // Find all non-redundant network input to neuron connections
    let valid_connections = network.neuron_ids().flat_map(|parent_id| {
        // The network input IDs that aren't connected to this neuron
        let mut missing_input_connections = HashSet::new();
        missing_input_connections.extend(0..individual.inputs);

        for g in get_direct_children(network, parent_id) {
            if let Gene::Input(input) = g {
                missing_input_connections.remove(&input.id().as_usize());
            }
        }

        missing_input_connections
            .into_iter()
            .map(move |input_id| (parent_id, InputId::new(input_id)))
    });

    // Choose one at random and add it
    if let Some((parent, id)) = valid_connections.choose(rng) {
        let input = Input::new(id, INITIAL_WEIGHT_VALUE);
        add_non_neuron(individual, parent, input);
        true
    } else {
        false
    }
}

/// Adds a subnetwork to a random parent neuron in the network. Randomly connects network inputs and
/// other neurons to it as inputs, and randomly connects its output to other neurons.
///
/// The original paper was not clear about how forward jumper connections should be added here. This
/// function simply applies a fixed probability to each possible connection, which results in more
/// connections being added on average to larger networks than to smaller ones.
fn add_subnetwork<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    rng: &mut ThreadRng,
) -> bool {
    let network = &mut individual.network;

    // Choose a random parent neuron to add the subnetwork to
    let parent = network.neuron_ids().choose(rng).unwrap();

    // Add random inputs to the subnetwork
    let mut subnetwork_inputs = Vec::new();

    // Each network input has a 50% chance of being connected
    for i in 0..individual.inputs {
        if rng.gen() {
            let input = Input::new(InputId::new(i), INITIAL_WEIGHT_VALUE);
            subnetwork_inputs.push(input.into());
        }
    }

    // Each neuron with greater depth also has a chance of being connected
    let parent_depth = network[parent].depth();
    let subnetwork_depth = parent_depth + 1;

    for id in network.get_valid_forward_jumper_sources(subnetwork_depth) {
        if rng.gen::<f64>() < NEW_SUBNETWORK_FORWARD_CONNECTION_PROBABILITY {
            let forward = ForwardJumper::new(id, INITIAL_WEIGHT_VALUE);
            subnetwork_inputs.push(forward.into());
        }
    }

    // If the subnetwork has no inputs, connect it to a random network input
    if subnetwork_inputs.is_empty() {
        if individual.inputs > 0 {
            let id = rng.gen_range(0..individual.inputs);
            let input = Input::new(InputId::new(id), INITIAL_WEIGHT_VALUE);
            subnetwork_inputs.push(input.into());
        } else {
            // If the network has no inputs, no subnetwork is added
            // It may be possible to find a different parent or input connection to add, but this case should be
            // extremely rare anyways
            return false;
        }
    }

    // Insert new age counters for the subnetwork's genes
    let parent_index = network[parent].subgenome_range().start;
    let subgenome_index = parent_index + 1;
    // Each element of `subnetwork_inputs` is a new gene, plus the subnetwork's root neuron itself
    let num_new_genes = 1 + subnetwork_inputs.len();
    individual.ages.splice(
        subgenome_index..subgenome_index,
        iter::repeat(0).take(num_new_genes),
    );

    // Add the subnetwork
    let subnetwork_id = network
        .add_subnetwork(parent, INITIAL_WEIGHT_VALUE, subnetwork_inputs)
        .unwrap();

    // Finally, the new subnetwork itself has a chance to be connected to each neuron with lesser
    // depth other than its parent
    let output_connections = network
        .neuron_info_map()
        .iter()
        .filter(|(id, info)| **id != parent && info.depth() < subnetwork_depth)
        .map(|(id, _)| *id)
        .filter(|_| rng.gen::<f64>() < NEW_SUBNETWORK_FORWARD_CONNECTION_PROBABILITY)
        .collect::<Vec<_>>();

    for id in output_connections {
        if rng.gen::<f64>() < NEW_SUBNETWORK_FORWARD_CONNECTION_PROBABILITY {
            let forward = ForwardJumper::new(subnetwork_id, INITIAL_WEIGHT_VALUE);
            add_non_neuron(individual, id, forward);
        }
    }

    true
}

/// Randomly adds a bias gene to the network. Returns whether any mutation was performed.
fn add_bias<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    rng: &mut ThreadRng,
) -> bool {
    let network = &mut individual.network;

    // Choose a random neuron without an existing bias input
    let valid_parents = network
        .neuron_ids()
        .filter(|id| get_direct_children(network, *id).all(|g| !g.is_bias()));
    let parent = valid_parents.choose(rng);

    // Add a bias gene to it
    if let Some(id) = parent {
        let bias = Bias::new(INITIAL_WEIGHT_VALUE);
        add_non_neuron(individual, id, bias);
        true
    } else {
        false
    }
}

/// Randomly removes a connection from the network. Returns whether any mutation was performed.
///
/// The original paper was not clear about whether bias and input genes count as connections for the
/// purposes of this mutation, but this function assumes they do and therefore may remove them. Note
/// that this may theoretically result in a network losing all connections to an input.
fn remove_connection<T: FitnessFunction + Clone>(
    individual: &mut Individual<T>,
    rng: &mut ThreadRng,
) -> bool {
    let network = &mut individual.network;

    // Choose a random, valid non-neuron to remove and remove it
    let index = network.get_valid_removals().choose(rng);
    if let Some(i) = index {
        network.remove_non_neuron(i).unwrap();
        individual.ages.remove(i);
        true
    } else {
        false
    }
}

/// Adds a non-neuron gene to the network and initializes a corresponding gene age counter.
fn add_non_neuron<T: FitnessFunction + Clone, G: Into<NonNeuronGene<f64>>>(
    individual: &mut Individual<T>,
    parent: NeuronId,
    gene: G,
) {
    let parent_index = individual.network[parent].subgenome_range().start;
    let gene_index = parent_index + 1;
    individual.network.add_non_neuron(parent, gene).unwrap();
    // Insert a new age counter for the gene
    individual.ages.insert(gene_index, 0);
}

/// Returns an iterator over the direct children of the neuron with the given ID, or panics if it
/// does not exist.
fn get_direct_children(network: &Network, id: NeuronId) -> impl Iterator<Item = &Gene<f64>> {
    // TODO: Subnetworks could be detected and skipped here, though it might not be much faster on
    //       average
    let range = network[id].subgenome_range();
    range.filter_map(move |i| {
        if network.parent_of(i).unwrap() == Some(id) {
            Some(&network[i])
        } else {
            None
        }
    })
}
