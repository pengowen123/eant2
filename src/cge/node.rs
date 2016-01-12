// An enumeration with each variant being a variant of a type, used in place of inheritance
// A vector of nodes is stored as a genome in a network

#[derive(Clone, Copy)]
pub struct Neuron {
    // Neurons receive inputs and create outputs
    // The input_count, depth, and id_number are calculated with a method from Genome and are
    // not received as inputs for creating a new Neuron
    pub current_value: f64, // For storing data while other things are calculated before moving on
    pub weight: f64, // The output of the neuron is multiplied by the weight
    pub input_count: i32, // The amount of inputs to the neuron
    pub id_number: i32, // The identification number of the neuron
}

#[derive(Clone, Copy)]
pub struct Input {
    // Inputs only create outputs, the id_number is calculated
    pub current_value: f64,
    pub weight: f64,
    pub id_number: i32,
}

#[derive(Clone, Copy)]
pub struct JumperForward {
    // Because of the encoding, neurons can only have one output. Jumpers let neurons have
    // multiple outputs. The jumper's output is implicit and the input is stored in id_number.
    pub weight: f64,
    pub id_number: i32, // The id_number of the neuron the connection is an input to
}

#[derive(Clone, Copy)]
pub struct JumperRecurrent {
    // JumperForward connects a neuron to one of higher depth, while JumperRecurrent does the opposite
    pub weight: f64,
    pub id_number: i32,
}

#[derive(Clone, Copy)]
pub enum Node {
    Neuron(Neuron),
    Input(Input),
    JumperForward(JumperForward),
    JumperRecurrent(JumperRecurrent),
}

impl Node {
    // Constructors
    pub fn new_neuron() -> Node {
        Node::Neuron(Neuron {
            current_value: 0.0,
            weight: 1.0,
            input_count: 1,
            id_number: 0,
        })
    }

    pub fn new_input() -> Node {
        Node::Input(Input {
            current_value: 0.0,
            weight: 1.0,
            id_number: 1,
        })
    }

    pub fn new_jumper_f() -> Node {
        Node::JumperForward(JumperForward {
            weight: 1.0,
            id_number: 0,
        })
    }

    pub fn new_jumper_r() -> Node {
        Node::JumperRecurrent(JumperRecurrent {
            weight: 1.0,
            id_number: 0,
        })
    }
}