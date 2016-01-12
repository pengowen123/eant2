use cge::network::Network;
use cge::node::Node;

pub struct NetworkCMAES {
    pub network: Network,
    pub fitness: f64
}


impl NetworkCMAES {
    pub fn convert(vec: &Vec<Network>) -> Vec<NetworkCMAES> {
        let mut converted = Vec::new();

        for element in vec {
            converted.push(NetworkCMAES {
                network: element.clone(),
                fitness: 0.0
            });
        }

        converted
    }
    
    pub fn set_parameters(network: &mut NetworkCMAES) {
    	for node in &mut network.network.genome {
    		
    	}
    }
}
