#[derive(Clone, Debug)]
pub struct NetworkParameters {
    pub parameters: Vec<f64>,
    pub fitness: f64
}

impl NetworkParameters {
    pub fn new(params: Vec<f64>) -> NetworkParameters {
        NetworkParameters {
            parameters: params,
            fitness: 0.0
        }
    }
}
