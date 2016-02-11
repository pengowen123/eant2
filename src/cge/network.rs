use std::ops::Range;

use cge::utils::Stack;
use cge::gene::*;

/// The neural network struct.
#[derive(Clone)]
pub struct Network {
    genome: Vec<Gene>
}

impl Network {
    /// The evaluation function.
    ///
    pub fn evaluate(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        vec![0.0]
    }

    fn evaluate_slice(&mut self, range: Range<usize>) -> Vec<f64> {
        self.genome[range].iter().map(|gene| gene.weight).collect()
    }
}
