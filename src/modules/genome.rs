use modules::functions::*;
use modules::stack;
use modules::node::*;

struct Genome {
    size: i32,
    fitness: f64,
    id_number: i32,
    age: i32,
    parents: Vec<i32>,
    chromosome: Vec<Box<NodeT>>
}

impl Genome {
    fn new() -> Genome {
        Genome {
            size: 0,
            fitness: -2_000_000_000.0,
            id_number: 0,
            age: 0,
            parents: Vec::new(),
            chromosome: Vec::new()
        }
    }

    fn add_node<T: 'static + Clone + NodeT>(&mut self, node: T) {
        // fix this by using Vec<Box<T>>
        // let node_type = type_of(&node);

        // if node_type == "Node" {}
        self.chromosome.push(Box::new(node.clone()));
        self.size = self.chromosome.len() as i32;
    }
    // continue adding functions here

    fn remove_node(&mut self, index: i32) {
        self.chromosome.remove(index as usize);
    }

    fn insert_node<T: 'static + Clone + NodeT>(&mut self, index: i32, node: Node) {
        self.chromosome.insert(index as usize, Box::new(node.clone()));
    }

    fn insert_genome(&mut self, index: i32, genome: Genome) {
        let mut i = index as usize;
        let mut n = 0usize;

        while i < genome.chromosome.len() {
            self.chromosome.insert(i.clone(), genome.chromosome[n].clone());
            i += 1usize;
            n += 1usize;
        }
    }

    fn delete_nodes(&mut self, start: i32, end: i32) {}
    fn delete_subnetwork(&mut self, index: i32) {}
    fn get_neuron_count(&self) /* -> i32 */ {}
    fn get_subnetwork(&self, index: i32) /* -> Genome */ {}
    fn get_neuron(&self, index: i32) /* -> Node */ {}
    fn reset_inputs(&mut self) {}
    fn update_depths(&mut self) {}
    fn next_subnetwork_index(&self, index: i32) /* -> i32 */ {}

}
