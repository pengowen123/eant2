use crate::cge_utils::Mutation;
use crate::utils::Individual;
use crate::FitnessFunction;
use cge::gene::GeneExtras;
use rand::{prelude::ThreadRng, Rng};
use std::ops;

// A few convenience methods for helping with determining which mutation operators are valid
impl<T: FitnessFunction + Clone> Individual<T> {
    // Returns the amount of connections from an input with the given id
    pub fn get_input_copies(&self, id: usize) -> usize {
        self.network.genome.iter().fold(0, |acc, g| {
            if let GeneExtras::Input(_) = (*g).variant {
                acc + ((g.id == id) as usize)
            } else {
                acc
            }
        })
    }

    // Returns a vector with each element being the length of the shortest path between the
    // corresponding neuron and the nearest output
    pub fn get_depths(&self, include_connections: bool) -> Vec<usize> {
        let mut depths = Vec::new();
        let mut stack = Vec::new();

        for gene in &self.network.genome {
            let depth = stack.len();

            if let GeneExtras::Neuron(_, ref inputs) = gene.variant {
                depths.push(depth);
                stack.push(*inputs);
            } else {
                if include_connections {
                    depths.push(depth);
                }
                while let Some(&1) = stack.last() {
                    stack.pop();
                }
                if let Some(last) = stack.last_mut() {
                    *last -= 1;
                }
            }
        }

        depths
    }

    pub fn random_index(&self, rng: &mut ThreadRng) -> usize {
        let nth = rng.gen_range(0..self.next_id);
        self.network.get_neuron_index(nth).unwrap()
    }

    pub fn subnetwork_index(&self, index: usize) -> ops::Range<usize> {
        let mut i = index;
        let mut sum = 0;

        while sum != 1 {
            if let GeneExtras::Neuron(_, ref inputs) = self.network.genome[i].variant {
                sum += 1 - *inputs as i32;
            } else {
                sum += 1;
            }

            i += 1;
        }

        ops::Range {
            start: index,
            end: i,
        }
    }
}

// Wrap the Network implementation, to adjust the gene_ages field as well as the genome
impl<T: FitnessFunction + Clone> Mutation for Individual<T> {
    // Inputs and outputs aren't used; read from field instead
    fn add_subnetwork(&mut self, _: usize, output: usize, _: usize) {
        self.network
            .add_subnetwork(self.next_id, output, self.inputs);

        if let GeneExtras::Neuron(_, ref inputs) = self.network.genome[output].variant {
            for _ in 0..*inputs + 1 {
                self.ages.insert(output, 0);
            }
        }

        self.next_id += 1;
    }

    fn add_forward(&mut self, input: usize, output: usize) {
        self.network.add_forward(input, output);
        // Add an age associated with the new connection
        self.ages.insert(output, 0);
    }

    fn add_recurrent(&mut self, input: usize, output: usize) {
        self.network.add_recurrent(input, output);
        // Add an age associated with the new connection
        self.ages.insert(output, 0);
    }

    fn add_bias(&mut self, output: usize) {
        self.network.add_bias(output);
        // Add an age associated with the new connection
        self.ages.insert(output, 0);
    }

    fn add_input(&mut self, input: usize, output: usize) {
        self.network.add_input(input, output);
        // Add an age associated with the new connection
        self.ages.insert(output, 0);
    }

    fn remove_connection(&mut self, index: usize, output: usize) {
        self.network.remove_connection(index, output);
        // Remove the age associated with the removed connection
        self.ages.remove(index);
    }

    // does not need to be implemented
    fn previous_neuron_index(&self, _: usize) -> Option<usize> {
        unimplemented!();
    }
}
