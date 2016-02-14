//! The neural network struct.
//!
//! # Examples
//!
//! ```
//! // Load a neural network from a file
//! let network = cge::load_from_file("neural_network.ann").unwrap();
//! 
//! // Get the output of the neural network with the specified inputs
//! let result = network.evaluate(vec![1.0, 1.0]);
//!
//! // Reset the neural network
//! network.clear_state();
//! ```
use std::ops::Range;
use std::path::Path;
use std::fs::File;

use cge::utils::Stack;
use cge::gene::*;
use cge::gene::GeneExtras::*;

const BIAS_GENE_VALUE: f64 = 1.0;

#[derive(Clone)]
pub struct Network {
    pub size: usize,
    pub genome: Vec<Gene>
}

impl Network {
    /// Evaluates the neural network with the given inputs, returning a vector of outputs. The encoding can
    /// encode recurrent connections and bias inputs, so an internal state is used. It is important to run
    /// the clear_state method before calling evaluate again, unless it is desired to allow data
    /// carry over from the previous evaluation, for example if the network is being used as a real
    /// time controller.
    /// 
    /// If too little inputs are given, the empty inputs will have a value of zero. If too many
    /// inputs are given, the extras are discarded.
    ///
    /// # Examples
    ///
    /// ```
    /// // Get the output of the neural network the the specified inputs
    /// let result = network.evaluate(vec![1.0, 1.0]);
    ///
    /// // Get the output of the neural network with no inputs
    /// let result = network.evaluate(Vec::new());
    ///
    /// // Get the output of the neural network with too many inputs (extras aren't used)
    /// let result = network.evaluate(vec![1.0, 1.0, 1.0]);
    ///
    /// 
    /// // Let's say adder.ann is a file with a neural network with recurrent connections, used for
    /// // adding numbers together.
    /// let adder = cge::load_from_file("adder.ann");
    ///
    /// // result_one will be 1.0
    /// let result_one = adder.evaluate(vec![1.0]);
    ///
    /// // result_two will be 3.0
    /// let result_two = adder.evaluate(vec![2.0]);
    ///
    /// // result_three will be 5.0
    /// let result_three = adder.evaluate(vec![2.0]);
    ///
    /// // If this behavior is not desired, call the clear_state method between evaluations:
    /// let result_one = adder.evaluate(vec![1.0]);
    /// 
    /// adder.clear_state();
    ///
    /// // The 1.0 from the previous call is gone, so result_two will be 2.0
    /// let result_two = adder.evaluate(vec![2.0]);
    /// ```
    pub fn evaluate(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        // Set inputs
        for gene in &mut self.genome {
            if let Input(ref mut current_value, _) = gene.variant {
                *current_value = match inputs.get(gene.id) {
                    Some(v) => *v,
                    None => 0.0
                }
            }
        }

        let size = self.size;
        self.evaluate_slice(0..size)
    }

    /// Clears the internal state of the neural network.
    pub fn clear_state(&mut self) {
        for gene in &mut self.genome {
            match gene.variant {
                Input(ref mut current_value, _) => {
                    *current_value = 0.0;
                },
                Neuron(ref mut current_value, _, _) => {
                    *current_value = 0.0;
                }
                _ => {}
            }
        }
    }

    /// Saves the neural network to a file.
    /// In the future the file format should be documented.
    pub fn save_to_file(&self) {

    }

    /// Loads a neural network from a file. The data is not validated, so it is up to the user to
    /// guarantee it is valid. If the network is invalid, there may be generic
    /// panic messages or invalid output.
    pub fn load_from_file() -> Option<Network> {
        // Make sure there is no NaN or infinite
        Some(Network { size: 0, genome: Vec::new() })
    }

    // Returns the output of sub-linear genome in the given range
    fn evaluate_slice(&mut self, range: Range<usize>) -> Vec<f64> {
        let mut i = range.end;
        // Initialize a stack for evaluating the neural network
        let mut stack = Stack::new();
        
        // Iterate backwards over the specified slice
        while i >= range.start {
            let variant = self.genome[i].variant.clone();

            match variant {
                Input(_, _) => {
                    // If the gene is an input, push its value multiplied by the inputs weight onto
                    // the stack
                    let (weight, id, value, outputs) = self.genome[i].ref_input().unwrap();
                    stack.push(weight * value);
                },
                Neuron(_, _, _) => {
                    // If the gene is a neuron, pop a number (the neurons input count) of inputs
                    // off the stack, and push their sum multiplied by the neurons weight onto the
                    // stack
                    let (weight, id, value, inputs, outputs) = self.genome[i].ref_mut_neuron().unwrap();

                    *value = stack.pop(*inputs).unwrap().iter().fold(0.0, |acc, i| acc + i);
                    stack.push(*weight * *value);
                },
                Forward => {
                    // This is inefficient because it can run the neuron evaluation code multiple
                    // times
                    // TODO: Turn current value of neurons into a struct with a flag for whether
                    // the neuron has been evaluated this network evaluation. Reset the flag every
                    // network evaluation.

                    // If the gene is a forward jumper, evaluate the subnetwork starting at the
                    // neuron with id of the jumper, and push the result multiplied by the jumpers
                    // weight onto the stack
                    let weight = self.genome[i].weight;
                    let id = self.genome[i].id;
                    let subnetwork_range = self.get_subnetwork_index(id).unwrap();

                    let result = self.evaluate_slice(subnetwork_range);
                    stack.push(weight * result[0]);
                },
                Recurrent => {
                    // If the gene is a recurrent jumper, push the previous value of the neuron
                    // with the id of the jumper multiplied by the jumpers weight onto the stack
                    let gene = &self.genome[i];
                    let neuron = &self.genome[self.get_neuron_index(gene.id).unwrap()];
                    
                    if let Neuron(ref current_value, _, _) = neuron.variant {
                        stack.push(gene.weight * *current_value);
                    }
                },
                Bias => {
                    // If the gene is a bias input, push the bias constant multiplied by the genes
                    // weight onto the stack
                    let gene = &self.genome[i];
                    stack.push(gene.weight * BIAS_GENE_VALUE);
                }
            }

            if i == range.start {
                break;
            }

            i -= 1;
        }

        stack.data
    }

    // Returns the start and end index of the subnetwork starting at the neuron with the given id,
    // or None if it does not exist
    fn get_subnetwork_index(&self, id: usize) -> Option<Range<usize>> {
        let start = match self.get_neuron_index(id) {
            Some(i) => i,
            None => return None
        };

        // end begins at start + 1 because of how ranges work
        let mut end = start + 1;
        let mut sum = if let Neuron(_, _, ref outputs) = self.genome[start].variant {
            *outputs
        } else {
            // Unreachable because if there is no neuron at the index, the function would have
            // returned None
            unreachable!();
        };

        // Iterate through genes after the start index, modifying the sum each step 
        // I could use an iterator here, but it would be messy
        for gene in &self.genome[start..self.size] {
            match gene.variant {
                Neuron(_, ref inputs, ref outputs) => {
                    sum += outputs - inputs;
                },
                _ => {
                    sum += 1;
                }
            }

            if sum == 1 {
                break;
            }

            end += 1;
        }

        if sum != 1 {
            None
        } else {
            Some(Range {
                start: start,
                end: end
            })
        }
    }

    // Returns the index of the neuron with the given id, or None if it does not exist
    fn get_neuron_index(&self, id: usize) -> Option<usize> {
        let mut result = None;

        for (i, gene) in self.genome.iter().enumerate() {
            if let Neuron(_, _, _) = gene.variant {
                if gene.id == id {
                    result = Some(i);
                }
            }
        }

        result
    }
}
