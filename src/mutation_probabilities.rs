use crate::mutation::MutationType;
use rand::prelude::Rng;
use rand_distr::{Distribution, WeightedAliasIndex, WeightedError};
use std::convert::TryFrom;

/// Describes the relative probabilites of different structural mutations.
///
/// - Use `MutationProbabilities` to create a `MutationSampler`.
#[derive(Clone)]
pub struct MutationSampler(WeightedAliasIndex<u16>);

impl MutationSampler {
    /// Table of `::sample()` output values.
    const OUTPUTS: [MutationType; 4] = [
      MutationType::AddConnection,
      MutationType::RemoveConnection,
      MutationType::AddNode,
      MutationType::AddBias
    ];

    fn assemble(probabilities: (u16, u16, u16, u16)) -> Result<Self, WeightedError> {
        let sampler = WeightedAliasIndex::new(vec![
            probabilities.0,
            probabilities.1,
            probabilities.2,
            probabilities.3,
        ])?;

        Ok(MutationSampler(sampler))
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> MutationType {
        let index = self.0.sample(rng);

        // soundness proof: 
        //   the probability list is only 4 elements long, see `::assemble`.
        //   the returned index is guaranteed to fall in [0, 3].
        unsafe { *Self::OUTPUTS.get_unchecked(index) }
    }
}

impl Default for MutationSampler {
  fn default() -> Self {
    // safe unwrap. infallible because of default parameter choice.
    MutationSampler::assemble((3, 8, 1, 3)).unwrap()
  }
}

/// Easier to use mutation probabilities.
/// - Input is a tuple of probabilities for each mutation.
/// - Automatically normalized upon conversion to `MutationSampler`.
#[derive(Copy, Clone)]
pub struct MutationProbabilities((f64, f64, f64, f64));

impl TryFrom<MutationProbabilities> for MutationSampler {
    type Error = WeightedError;

    fn try_from(p: MutationProbabilities) -> Result<Self, WeightedError> {
        let p = p.0;
        let sum = p.0 + p.1 + p.2 + p.3;

        // map to u16s with maximum possible precision
        let scaling = (u16::MAX as f64) / sum;

        MutationSampler::assemble((
            (p.0 * scaling) as u16,
            (p.1 * scaling) as u16,
            (p.2 * scaling) as u16,
            (p.3 * scaling) as u16,
        ))
    }
}

impl MutationProbabilities {
    /// All probabilities start as zero.
    /// - You must provide at least one non-zero probability.
    /// - No probability may be negative or `f64::NAN`.
    pub const fn zeros() -> Self {
        MutationProbabilities((0., 0., 0., 0.))
    }

    /// Sugar over `try_from`. 
    /// Will fail if any probability is negative, all are zero, or any is `f64::NAN`.
    pub fn build(self) -> Result<MutationSampler, WeightedError> {
      MutationSampler::try_from(self)
    }
}

impl MutationProbabilities {
    /// The probability of adding a new connection between neurons.  Represented as the relative size of this number to the others.
    pub const fn add_connection(mut self, probability: f64) -> Self {
        self.0 .0 = probability;
        self
    }

    /// The probability of removing an existing connection between neurons.  Represented as the relative size of this number to the others.
    pub const fn remove_connection(mut self, probability: f64) -> Self {
        self.0 .1 = probability;
        self
    }

    /// The probability of adding a new neuron.  Represented as the relative size of this number to the others.
    pub const fn add_neuron(mut self, probability: f64) -> Self {
        self.0 .2 = probability;
        self
    }

    /// The probability of adding a new bias neuron. Represented as the relative size of this number to the others.
    pub const fn add_bias(mut self, probability: f64) -> Self {
        self.0 .3 = probability;
        self
    }
}
