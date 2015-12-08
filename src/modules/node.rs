#![allow(dead_code)]
#![allow(unused_variables)]

#[derive(Clone, Copy)]
pub struct Neuron {
    // Neurons receive inputs and create outputs
    // The input_count, depth, and id_number are calculated with a method from Genome and are
    // not received as inputs for creating a new Neuron
    pub current_value: f64, // For storing data while other things are calculated before moving on
    pub weight: f64, // The output of the neuron is multiplied by the weight
    pub depth: i32, // The amount of neurons between this one and the output of the network
    pub input_count: i32, // The amount of inputs to the neuron
    pub id_number: i32 // The identification number of the neuron
}

#[derive(Clone, Copy)]
pub struct Input {
    // Inputs only create outputs, the id_number is calculated
    pub current_value: f64,
    pub weight: f64,
    pub id_number: i32
}

#[derive(Clone, Copy)]
pub struct JumperF {
    // Because of the encoding, neurons can only have one output. Jumpers let neurons have
    // multiple outputs. The jumper's input is implicit and the output is stored as a field.
    pub current_value: f64,
    pub weight: f64,
    pub id_number: i32 // The id_number of the neuron the connection is an input to
}

#[derive(Clone, Copy)]
pub struct JumperR {
    // JumperF connects a neuron to one of higher depth, while JumperR does the opposite
    pub current_value: f64,
    pub weight: f64,
    pub id_number: i32
}

#[derive(Clone, Copy)]
pub enum Node {
    Neuron(Neuron),
    Input(Input),
    JumperF(JumperF),
    JumperR(JumperR)
}

impl Node {
    pub fn new_neuron() -> Node {
        Node::Neuron(Neuron {
            current_value: 0.0,
            weight: 1.0,
            depth: 0,
            input_count: 1,
            id_number: 0
        })
    }

    pub fn new_input() -> Node {
        Node::Input(Input {
            current_value: 0.0,
            weight: 1.0,
            id_number: 1
        })
    }

    pub fn new_jumper_f() -> Node {
        Node::JumperF(JumperF {
            current_value: 0.0,
            weight: 1.0,
            id_number: 0
        })
    }

    pub fn new_jumper_r() -> Node {
        Node::JumperR(JumperR {
            current_value: 0.0,
            weight: 1.0,
            id_number: 0
        })
    }
}
