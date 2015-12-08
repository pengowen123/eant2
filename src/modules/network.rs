#![allow(dead_code)]
#![allow(unused_variables)]

use modules::functions::*;
use modules::stack::*;
use modules::node::*;

pub struct Network {
    // The size, id_number, and parents of a genome are calculated and are not received as inputs
    // for creating a Genome
    pub size: i32, // The amount of nodes in the genome
    pub id_number: i32, // The identification number of the genome
    pub parents: Vec<i32>, // A vector containing the id_number's of its parents
    pub genome: Vec<Node>, // A vector containing the nodes of the genome
    pub state: Vec<Node>
}

impl Network {
    // instead of evaluate() use step() where step() updates the internal state, returning outputs
    // and taking new inputs each time step
    pub fn new() -> Network {
        Network {
            size: 0,
            id_number: 0,
            parents: Vec::new(),
            genome: Vec::new(),
            state: Vec::new()
        }
    }

    pub fn step(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        // make better step function, this is really bad
        let mut stack = Stack::new();
        let mut genome = reverse(&self.genome);

        let mut current_input = 0;
        let mut i = 0;

        while i < genome.len() {
            if current_input > inputs.len() {
                break;
            }

            let element = &mut genome[i];
            match *element {
                Node::Input(Input {
                    ref mut current_value,
                    ref mut weight,
                    ref mut id_number
                }) => {
                    *current_value = inputs[current_input];
                    current_input += 1;
                },

                _ => {}
            }
            i += 1;
        }

        // Bias input
        // Increment current_value of the first input
        i = 0;
        while i < genome.len() {

            let element = &mut genome[i];
            match *element {
                Node::Input(Input {
                    ref mut current_value,
                    ref mut weight,
                    ref mut id_number
                }) => *current_value += 1.0,

                _ => {}
            }
            i += 1;
        }

        for element in genome {
            println!("{:?}", stack.vec);
            match element {
                Node::Neuron(x) => {
                    let inputs = stack.pop(x.input_count.clone());
                    stack.push(sum_vec(&inputs) * x.weight)
                },

                Node::Input(x) => stack.push(x.current_value * x.weight),

                _ => unreachable!()
            }
        }

        stack.vec
    }

    fn update_depths(&mut self) {}
    fn update_ids(&mut self) {}
}
