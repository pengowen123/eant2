/// An enum for storing additional information for different types of genes
#[derive(Clone)]
pub enum GeneExtras {
    /// Input contains a current value and an output count
    Input(f64, usize),
    /// Neuron contains a current value, an input count and an output count
    Neuron(f64, usize, usize),
    Forward,
    Recurrent,
    Bias
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
            variant: GeneExtras::Input(0.0, 0)
        }
    }

    pub fn bias(weight: f64) -> Gene {
        Gene {
            weight: weight,
            id: 0,
            variant: GeneExtras::Bias
        }
    }

    pub fn neuron(weight: f64, id: usize, inputs: usize) -> Gene {
        Gene {
            weight: weight,
            id: id,
            variant: GeneExtras::Neuron(0.0, inputs, 0)
        }
    }

    pub fn ref_input(&self) -> Option<(f64, usize, f64, usize)> {
        if let GeneExtras::Input(ref weight, ref outputs) = self.variant {
            Some((self.weight, self.id, *weight, *outputs))
        } else {
            None
        }
    }

    pub fn ref_neuron(&self) -> Option<(f64, usize, f64, usize, usize)> {
        if let GeneExtras::Neuron(ref value, ref inputs, ref outputs) = self.variant {
            Some((self.weight, self.id, *value, *inputs, *outputs))
        } else {
            None
        }
    }

    pub fn ref_mut_neuron<'a>(&'a mut self) -> Option<(&'a mut f64, &'a mut usize, &'a mut f64, &'a mut usize, &'a mut usize)> {
        if let GeneExtras::Neuron(ref mut value, ref mut inputs, ref mut outputs) = self.variant {
            Some((&mut self.weight, &mut self.id, value, inputs, outputs))
        } else {
            None
        }
    }
}
