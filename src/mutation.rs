use cge::gene::GeneExtras;
use rand::{Rng, thread_rng};
use crate::mutation_probabilities::MutationProbabilities;
use crate::utils::Individual;
use crate::cge_utils::Mutation;
use crate::FitnessFunction;

/// The type of mutation to perform.
pub enum MutationType {
  AddConnection,
  RemoveConnection,
  AddNode,
  AddBias
}

/// Selects a random valid mutation and applies it to a neural network
pub fn mutate<T: FitnessFunction + Clone>(individual: &mut Individual<T>, probabilities: MutationProbabilities) {
  let mut rng = thread_rng();

  // index of the neuron to make the new mutation's output lead to
  let index = individual.random_index() + 1;
  let depths = individual.get_depths(true);
  let neuron_depth = depths[index - 1];

  match probabilities.generate(&mut rng) {
    MutationType::AddConnection    => add_connection(individual, index, None),
    MutationType::AddNode          => individual.add_subnetwork(0, index, 0),
    MutationType::AddBias          => individual.add_bias(index),
    MutationType::RemoveConnection => {
      let valid_neurons = 
        individual
          .network
          .genome
          .iter()
          .filter(|g| {
            if let GeneExtras::Neuron(_, ref inputs) = g.variant { *inputs > 1 } 
            else { false }
          })
          .map(|n| n.id)
          .collect::<Vec<usize>>();

      if !valid_neurons.is_empty() {
        let id = rng.choose(&valid_neurons).expect("40");
        let depths = individual.get_depths(true);
        let range = individual.subnetwork_index(*id);
        let mut valid_indices = Vec::new();

        for i in range {
          let mut is_valid = true;
          let gene = &individual.network.genome[i];
          if depths[i] != neuron_depth + 1 { continue; }

          if let GeneExtras::Neuron(_, _) = gene.variant { continue; }
          if let GeneExtras::Input(_) = gene.variant { is_valid = individual.get_input_copies(gene.id) > 1 }
          if is_valid { valid_indices.push(i); }
        }

        if !valid_indices.is_empty() {
          let index = rng.choose(&valid_indices).expect("420");
          individual.remove_connection(*index, *id);
        }
      }
    }
  }
}

fn add_connection<T: FitnessFunction + Clone>(individual: &mut Individual<T>, index: usize, id_: Option<usize>) {
    let mut rng = thread_rng();
    let range = individual.subnetwork_index(index);
    let neuron_depth = individual.get_depths(true)[index];

    let mut valid_inputs = Vec::new();

    for i in 0..individual.inputs {
        let mut is_duplicate = false;

        for g in &individual.network.genome[range.clone()] {
            if let GeneExtras::Input(_) = g.variant {
                is_duplicate = true;
                break;
            }
        }

        if !is_duplicate {
            valid_inputs.push(i);
        }
    }

    let neurons = 0..individual.next_id;
    let depths_neurons = individual.get_depths(false);
    let id = match id_ {
        Some(i) => i,
        None => individual.network.genome[index - 1].id
    };

    let mut valid_forward = Vec::new();
    for i in neurons.clone() {
        if i == id {
            continue;
        }

        let mut is_duplicate = false;

        for g in &individual.network.genome[range.clone()] {
            if let GeneExtras::Forward = g.variant {
                if g.id == i {
                    is_duplicate = true;
                    break;
                }
            } else if let GeneExtras::Neuron(_, _) = g.variant {
                let other_range = individual.subnetwork_index(g.id);

                if range.start < other_range.start && range.end >= other_range.end {
                    is_duplicate = true;
                    break;
                }
            }
        }

        if depths_neurons[i] > neuron_depth && !is_duplicate {
            valid_forward.push(i);
        }
    }

    let mut valid_recurrent = Vec::new();
    for i in neurons.clone() {
        if i == id {
            continue;
        }

        let mut is_duplicate = false;

        for g in &individual.network.genome {
            if let GeneExtras::Recurrent = g.variant {
                if g.id == i {
                    is_duplicate = true;
                }
            }
        }
        
        if !is_duplicate {
            valid_recurrent.push(i);
        }
    }

    let mut valid_mutations = Vec::new();

    if !valid_inputs.is_empty() {
        valid_mutations.push(0);
    }

    if !valid_forward.is_empty() {
        valid_mutations.push(1);
    }

    if !valid_recurrent.is_empty() {
        valid_mutations.push(2);
    }

    if !valid_mutations.is_empty() {
        let mutation = rng.choose(&valid_mutations).expect("2");

        if let Some(v) = id_ {
            let mut is_valid = false;

            for i in &valid_forward {
                if *i == v {
                    is_valid = true;
                }
            }

            if is_valid {
                individual.add_forward(id, index);

                return;
            }
        }

        match *mutation {
            0 => {
                // Add input connection

                let input = rng.choose(&valid_inputs).expect("3");

                individual.add_input(*input, index);
            },
            1 => {
                // Add neuron connection
                
                let id = rng.choose(&valid_forward).expect("5");

                individual.add_forward(*id, index);
            },
            2 => {
                // Add recurrent neuron connection

                let id = rng.choose(&valid_recurrent).unwrap();

                individual.add_recurrent(*id, index);
            },
            _ => {
                unreachable!();
            }
        }
    }
}
