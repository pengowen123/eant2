pub trait NodeT {}

#[derive(Clone)]
pub enum NodeType {
    Neuron,
    Input,
    JForward,
    JRecurrent
}

#[derive(Clone)]
pub enum NodeAge {
    New,
    Old
}

#[derive(Clone)]
pub enum NeuronType {
    Output,
    Hidden
}

// I wish I could fix having so many implementations
// Each of these represents a variant of Node
// This holds information about the nodes that are stored in a chromosome vector

#[derive(Clone)]
pub struct Node {
    weight: f64,
    current_value: f64,
    learning_rate: f64,
    node_type: NodeType,
    node_age: NodeAge
}

#[derive(Clone)]
pub struct Neuron {
    weight: f64,
    current_value: f64,
    learning_rate: f64,
    node_type: NodeType,
    neuron_type: NeuronType,
    node_age: NodeAge,
    id_number: i32,
    input_count: i32,
    depth: i32,
    sigmoid_time: f64,
    sigmoid_rate: f64
}

#[derive(Clone)]
pub struct Input {
    weight: f64,
    current_value: f64,
    learning_rate: f64,
    node_type: NodeType,
    node_age: NodeAge,
    input_number: i32
}

#[derive(Clone)]
pub struct JForward {
    weight: f64,
    current_value: f64,
    learning_rate: f64,
    node_type: NodeType,
    node_age: NodeAge,
    id_number: i32
}

#[derive(Clone)]
pub struct JRecurrent {
    weight: f64,
    current_value: f64,
    learning_rate: f64,
    node_type: NodeType,
    node_age: NodeAge,
    id_number: i32
}


impl Node {
    pub fn new(weight: f64,
           current_value: f64,
           learning_rate: f64,
           node_type: NodeType,
           node_age: NodeAge) -> Node {
        Node {
            weight: weight,
            current_value: current_value,
            learning_rate: learning_rate,
            node_type: node_type,
            node_age: node_age
        }
    }

    pub fn new_empty() -> Node {
        Node {
            weight: 1.0,
            current_value: 0.0,
            learning_rate: 1.0,
            node_type: NodeType::Neuron,
            node_age: NodeAge::New
        }
    }
}

impl Neuron {
    pub fn new(weight: f64,
           current_value: f64,
           learning_rate: f64,
           neuron_type: NeuronType,
           node_age: NodeAge,
           id_number: i32,
           input_count: i32,
           depth: i32,
           sigmoid_time: f64,
           sigmoid_rate: f64) -> Neuron {
        Neuron {
            weight: weight,
            current_value: current_value,
            learning_rate: learning_rate,
            node_type: NodeType::Neuron,
            neuron_type: neuron_type,
            node_age: node_age,
            id_number: id_number,
            input_count: input_count,
            depth: depth,
            sigmoid_time: sigmoid_time,
            sigmoid_rate: sigmoid_rate
        }
    }

    pub fn new_empty() -> Neuron {
        Neuron {
            weight: 1.0,
            current_value: 0.0,
            learning_rate: 1.0,
            node_type: NodeType::Neuron,
            neuron_type: NeuronType::Hidden,
            node_age: NodeAge::New,
            id_number: 0,
            input_count: 1,
            depth: 1,
            sigmoid_time: 1.0,
            sigmoid_rate: 1.0
        }
    }
}

impl Input {
    pub fn new(weight: f64,
           current_value: f64,
           learning_rate: f64,
           node_type: NodeType,
           node_age: NodeAge,
           input_number: i32) -> Input {
        Input {
            weight: weight,
            current_value: current_value,
            learning_rate: learning_rate,
            node_type: node_type,
            node_age: node_age,
            input_number: input_number
        }
    }

    pub fn new_empty() -> Input {
        Input {
            weight: 1.0,
            current_value: 0.0,
            learning_rate: 1.0,
            node_type: NodeType::Input,
            node_age: NodeAge::New,
            input_number: 0
        }
    }
}

impl JForward {
    pub fn new(weight: f64,
           current_value: f64,
           learning_rate: f64,
           node_type: NodeType,
           node_age: NodeAge,
           id_number: i32) -> JForward {
        JForward {
            weight: weight,
            current_value: current_value,
            learning_rate: learning_rate,
            node_type: node_type,
            node_age: node_age,
            id_number: id_number
        }
    }

    pub fn new_empty() -> JForward {
        JForward {
            weight: 1.0,
            current_value: 0.0,
            learning_rate: 1.0,
            node_type: NodeType::JForward,
            node_age: NodeAge::New,
            id_number: 0
        }
    }
}

impl JRecurrent {
    pub fn new(weight: f64,
           current_value: f64,
           learning_rate: f64,
           node_type: NodeType,
           node_age: NodeAge,
           id_number: i32) -> JRecurrent {
        JRecurrent {
            weight: weight,
            current_value: current_value,
            learning_rate: learning_rate,
            node_type: node_type,
            node_age: node_age,
            id_number: id_number
        }
    }

    pub fn new_empty() -> JRecurrent {
        JRecurrent {
            weight: 1.0,
            current_value: 0.0,
            learning_rate: 1.0,
            node_type: NodeType::JRecurrent,
            node_age: NodeAge::New,
            id_number: 0
        }
    }
}

// allows Vec<Box<NodeT>> for mixed type vector
impl NodeT for Node {}
impl NodeT for Neuron {}
impl NodeT for Input {}
impl NodeT for JForward {}
impl NodeT for JRecurrent {}
