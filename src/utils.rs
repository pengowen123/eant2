use cge::Network;
use cge::gene::Gene;

pub struct Individual {
    pub network: Network,
    pub fitness: f64
}

pub struct GeneAge {
    pub gene: Gene,
    pub age: usize
}
