use cge::Network;
use cge::gene::GeneExtras::*;

pub trait NetworkUtils {
    fn update_output_counts(&mut self);
}

impl NetworkUtils for Network {
    // TODO: Replace this inefficient method with a shift register type system, the same as used
    // for updating input count
    fn update_output_counts(&mut self) {
        let mut mask = vec![None; self.genome.len()];

        for (i, gene) in self.genome.iter().enumerate() {
            match gene.variant {
                Input(_, _) => {
                    mask[i] = Some(self.genome.iter().fold(1, |acc, g| {
                        if let Input(_, _) = g.variant {
                            if g.id == gene.id {
                                acc + 1
                            } else {
                                acc
                            }
                        } else {
                            acc
                        }
                    }))
                },
                Neuron(_, _, _) => {
                    mask[i] = Some(self.genome.iter().fold(1, |acc, g| {
                        match g.variant {
                            Forward | Recurrent => {
                                if g.id == gene.id {
                                    acc + 1
                                } else {
                                    acc
                                }
                            },
                            _ => {
                                acc
                            }
                        }
                    }))
                }
                _ => {}
            }
        }

        for i in 0..self.genome.len() {
            match self.genome[i].variant {
                Input(_, ref mut outputs) | Neuron(_, _, ref mut outputs) => {
                    if let Some(v) = mask[i] {
                        *outputs = v;
                    }
                }
                _ => {}
            }
        }
    }
}
