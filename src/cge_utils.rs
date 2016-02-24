use rand::{thread_rng, Rng};
use cge::Network;
use cge::gene::{GeneExtras, Gene};

pub trait Mutation {
    fn add_subnetwork(&mut self, input: usize, output: usize, inputs: usize);
    fn add_forward(&mut self, input: usize, output: usize);
    fn add_recurrent(&mut self, input: usize, output: usize);
    fn add_bias(&mut self, output: usize);
    fn add_input(&mut self, input: usize, output: usize);
    fn remove_connection(&mut self, index: usize, output: usize);
    fn previous_neuron_index(&self, index: usize) -> Option<usize>;
}

impl Mutation for Network {
    // inputs is the number of inputs to the network
    // id is the id of the new neuron
    // output is the index to put the neuron at
    fn add_subnetwork(&mut self, id: usize, output: usize, inputs: usize) {
        let mut rng = thread_rng();
        let mut input_count = 0;

        for i in 0..inputs {
            if rng.gen() {
                self.genome.insert(output, Gene::input(1.0, i));
                input_count += 1;
            }
        }

        if input_count == 0 {
            self.genome.insert(output, Gene::input(1.0, rng.gen_range(0, inputs)));
            input_count = 1;
        }
        
        self.genome.insert(output, Gene::neuron(1.0, id, input_count));
        
        let prev_index = self.previous_neuron_index(output);

        if let Some(i) = prev_index {
            let (_, _, _, inputs) = self.genome[i].ref_mut_neuron().unwrap();

            *inputs += 1;
        }

        self.size = self.genome.len() - 1;
    }

    // input is the id of the neuron to take input from
    // output is the index to put the jumper at
    fn add_forward(&mut self, input: usize, output: usize) {
        self.genome.insert(output, Gene::forward(1.0, input));

        self.size = self.genome.len() - 1;

        let prev_index = self.previous_neuron_index(output).unwrap();
        let (_, _, _, inputs) = self.genome[prev_index].ref_mut_neuron().unwrap();

        *inputs += 1;
    }

    // input is the id of the neuron to take input from
    // output is the index to put the jumper at
    fn add_recurrent(&mut self, input: usize, output: usize) {
        self.genome.insert(output, Gene::recurrent(1.0, input));

        self.size = self.genome.len() - 1;

        let prev_index = self.previous_neuron_index(output).unwrap();
        let (_, _, _, inputs) = self.genome[prev_index].ref_mut_neuron().unwrap();

        *inputs += 1;
    }

    // input is the id of the input to add a connection from
    // output is the index to put the input connection at
    fn add_input(&mut self, input: usize, output: usize) {
        self.genome.insert(output, Gene::input(1.0, input));

        self.size = self.genome.len() - 1;

        let prev_index = self.previous_neuron_index(output).unwrap();
        let (_, _, _, inputs) = self.genome[prev_index].ref_mut_neuron().unwrap();

        *inputs += 1;
    }

    // output is the index to put the bias at
    fn add_bias(&mut self, output: usize) {
        self.genome.insert(output, Gene::bias(1.0));

        self.size = self.genome.len() - 1;

        let prev_index = self.previous_neuron_index(output).unwrap();
        let (_, _, _, inputs) = self.genome[prev_index].ref_mut_neuron().unwrap();

        *inputs += 1;
    }

    // output is the id of the neuron to remove the connection from
    // index is the index of the gene to remove
    fn remove_connection(&mut self, index: usize, output: usize) {
        self.genome.remove(index);

        self.size = self.genome.len() - 1;

        let neuron_index = match self.get_neuron_index(output) {
            Some(v) => v,
            None => {
                println!("{}", output);
                println!("{:?}", self);
                panic!();
            }
        };

        let (_, _, _, inputs) = self.genome[neuron_index].ref_mut_neuron().expect("bar");

        *inputs -= 1;
    }
    
    // get index of the first neuron before the index
    // to prevent code duplication
    fn previous_neuron_index(&self, index: usize) -> Option<usize> {
        for i in (0..index).rev() {
            if let GeneExtras::Neuron(_, _) = self.genome[i].variant {
                return Some(i);
            }
        }

        None
    }
}


