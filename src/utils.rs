use cge::Network;
use cge::gene::Gene;

#[derive(Clone)]
pub struct Individual {
    pub network: Network,
    pub fitness: f64,
    pub duplicates: usize,
    pub similar: usize
}

#[derive(Clone)]
pub struct GeneAge {
    pub gene: Gene,
    pub age: usize
}
