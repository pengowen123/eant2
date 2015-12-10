use modules::functions::*;
use modules::stack::*;
use modules::node::*;

const BIAS_INPUT: f64 = 1.0; //Allow this to be changeable in the future

pub struct Network {
    // The size, id_number, and parents of a genome are calculated and are not received as inputs
    // for creating a Genome
    pub size: i32, // The amount of nodes in the genome
    pub id_number: i32, // The identification number of the genome
    pub parents: Vec<i32>, // A vector containing the id_number's of its parents
    pub genome: Vec<Node> // A vector containing the nodes of the genome
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
        }
    }

    pub fn step(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        // Create a stack for pushing values onto
        // The genome is read right to left like Reverse Polish Notation
        let mut stack = Stack::new();
        let mut genome = reverse(&self.genome);
        let mut current_input = 0;
        let mut i = 0;

        // Reset inputs to 0
        while i < self.genome.len() {
            let element = &mut self.genome[i];
            match *element {
                Node::Input(Input {
                    ref mut current_value,
                    ref mut weight,
                    ref mut id_number
                }) => *current_value = 0.0,
                _ => {}
            }
            i += 1;
        }

        // Set inputs
        i = 0;
        while i < genome.len() {
            if current_input >= inputs.len() {
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
                }) => {
                    *current_value += BIAS_INPUT;
                    break;
                },

                _ => {}
            }
            i += 1;
        }

        i = 0;
        while i < genome.len() {

            let element = &mut genome[i];
            match *element {
                Node::Input(Input {
                    ref current_value,
                    ref weight,
                    ref id_number
                }) => stack.push(current_value * weight),

                Node::Neuron(Neuron {
                    ref mut current_value,
                    ref mut weight,
                    ref mut input_count,
                    ref mut id_number
                }) => {
                    *current_value = sum_vec(&stack.pop(*input_count));
                    stack.push(*current_value * *weight); // Why don't I need a semi-colon here?
                },

                Node::JumperF(JumperF {
                    ref mut weight,
                    ref mut id_number
                }) => {
                    let mut subnetwork = self.get_subnetwork(*id_number);
                    let result = subnetwork.step(vec![])[0];
                    stack.push(result * *weight);
                },

                Node::JumperR(JumperR {
                    ref mut weight,
                    ref mut id_number
                }) => {
                    let index = self.get_neuron_index(*id_number);
                    let previous_value = match self.genome[index] {
                        Node::Neuron(Neuron {
                            ref current_value,
                            ref weight,
                            ref input_count,
                            ref id_number
                        }) => current_value,
                        _ => unreachable!()
                    };
                    stack.push(previous_value * *weight);
                }
            }
            i += 1;
        }

        self.genome = reverse(&genome);
        stack.vec
    }

    fn get_neuron_index(&self, id: i32) -> usize {
        let mut index = 0usize;
        while index < self.genome.len() {
            match self.genome[index] {
                Node::Neuron(Neuron {
                    ref current_value,
                    ref weight,
                    ref input_count,
                    ref id_number
                }) => {
                    if id_number == &id {
                        break;
                    } else if index + 1 == self.genome.len() {
                        panic!("no neuron with id: {} was found", id);
                    }
                },
                _ => {}
            }
            index += 1usize;
        }
        index
    }

    fn get_subnetwork(&self, id: i32) -> Network {
        let mut sum = 0;
        let mut index = self.get_neuron_index(id);
        let mut genome = Vec::new();

        while index < self.genome.len() {
            genome.push(self.genome[index]);

            match self.genome[index] {
                Node::Neuron(Neuron {
                    ref current_value,
                    ref weight,
                    ref input_count,
                    ref id_number
                }) => sum += 1 - *input_count,
                _ => sum += 1
            }

            if sum == 1 {
                break;
            }

            index += 1usize;
        }

        Network {
            size: 0,
            id_number: 0,
            parents: Vec::new(),
            genome: genome
        }
    }

    // requires magic
    // figure this out soon
    fn update_input_count(&mut self) {
        let mut i = 0;
        while i < self.genome.len() {
            let element = &mut self.genome[i];
            match *element {
                Node::Neuron(Neuron {
                    ref mut current_value,
                    ref mut weight,
                    ref mut input_count,
                    ref mut id_number
                }) => {

                },
                _ => {}
            }
            i += 1;
        }
    }

    fn clear_genome(&mut self) {
        let mut i = 0;
        while i < self.genome.len() {
            let element = &mut self.genome[i];
            match *element {
                Node::Input(Input {
                    ref mut current_value,
                    ref mut weight,
                    ref mut id_number
                }) => *current_value = 0.0,

                Node::Neuron(Neuron {
                    ref mut current_value,
                    ref mut weight,
                    ref mut input_count,
                    ref mut id_number
                }) => *current_value = 0.0,
                _ => {}
            }
            i += 1;
        }
    }
}
