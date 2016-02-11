#[derive(Clone)]
pub enum GeneExtras {
    Input(f64),
    Neuron(f64, usize),
    Forward,
    Recurrent
}

#[derive(Clone)]
pub struct Gene {
    pub weight: f64,
    pub id: usize,
    pub variant: GeneExtras
}

impl Gene {
    pub fn forward(weight: f64, id: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Forward
        }
    }

    pub fn recurrent(weight: f64, id: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Recurrent
        }
    }

    pub fn input(weight: f64, id: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Input(0.0)
        }
    }

    pub fn neuron(weight: f64, id: usize, inputs: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Neuron(0.0, inputs)
        }
    }
}
