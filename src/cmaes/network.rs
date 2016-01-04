use cge::network::Network;

pub struct NetworkCMAES {
    pub network: Network,
    pub fitness: f64,
    pub thing: f64,
}


impl NetworkCMAES {
    pub fn convert(vec: &Vec<Network>) -> Vec<NetworkCMAES> {
        let mut converted = Vec::new();

        for element in vec {
            converted.push(NetworkCMAES {
                network: element.clone(),
                fitness: 0.0,
                thing: 0.0,
            });
        }

        converted
    }
}
