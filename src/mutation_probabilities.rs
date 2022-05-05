use crate::mutation::MutationType;
use rand::prelude::Rng;
use rand_distr::{Distribution, WeightedAliasIndex, WeightedError};
use std::convert::TryFrom;

/// Generates different structural mutations. Uses configurable relative probabilites.
///
/// - Use `MutationProbabilities` to create a `MutationSampler`:
/// ```rust
/// MutationProbabilities::zeros()
///   .add_connection(2.)     // 1/3rd chance = 2 / (2 + 2 + 1 + 1)
///   .remove_connection(2.)  // 1/3rd chance
///   .add_neuron(1.)         // 1/6th chance
///   .add_bias(1.)           // 1/6th chance
///   .build()?               // build the sampler
/// ```
#[derive(Clone)]
pub struct MutationSampler(WeightedAliasIndex<u16>);

impl MutationSampler {
    /// Table of `::sample()` output values.
    const OUTPUTS: [MutationType; 4] = [
        MutationType::AddConnection,
        MutationType::RemoveConnection,
        MutationType::AddNode,
        MutationType::AddBias,
    ];

    /// Create a new `MutationSampler` with given relative probabilities as u16 values.
    fn new(probabilities: (u16, u16, u16, u16)) -> Result<Self, WeightedError> {
        let sampler = WeightedAliasIndex::new(vec![
            probabilities.0,
            probabilities.1,
            probabilities.2,
            probabilities.3,
        ])?;

        Ok(MutationSampler(sampler))
    }

    /// Sample a mutation type (using relative probabilities declared on creation).
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
        MutationSampler::new((3, 8, 1, 3)).unwrap()
    }
}

/// Relative mutation probabilities.
/// - Probabilities automatically normalized upon conversion to `MutationSampler`.
/// - In general, the chance for connection mutations should be higher than the chance for a neuron mutation.
/// - If a minimal network is desired, set the bias, neuron and connection addition probabilities low,
///   and the connection removal probability high.
/// - For complex problems where network size isn't an issue, high connection addition probability is a good idea.
/// 
/// ```rust
/// MutationProbabilities::zeros()
///   .add_connection(2.)     // 1/3rd chance = 2 / (2 + 2 + 1 + 1)
///   .remove_connection(2.)  // 1/3rd chance
///   .add_neuron(1.)         // 1/6th chance
///   .add_bias(1.)           // 1/6th chance
///   .build()?               // build the sampler
/// ```
#[derive(Copy, Clone)]
pub struct MutationProbabilities((f64, f64, f64, f64));

impl TryFrom<MutationProbabilities> for MutationSampler {
    type Error = WeightedError;

    fn try_from(p: MutationProbabilities) -> Result<Self, WeightedError> {
        let MutationProbabilities((a, b, c, d)) = p;

        // map to u16s with maximum possible precision
        let scaling = {
            let sum = a + b + c + d;
            (u16::MAX as f64) / sum
        };

        MutationSampler::new((
            (a * scaling) as u16,
            (b * scaling) as u16,
            (c * scaling) as u16,
            (d * scaling) as u16,
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
    /// - Will fail if any probability is negative, all are zero, or any is `f64::NAN`.
    pub fn build(self) -> Result<MutationSampler, WeightedError> {
        MutationSampler::try_from(self)
    }
}

impl MutationProbabilities {
    /// The probability of adding a new connection between neurons.  Represented as the relative size of this number to the others.
    pub const fn add_connection(self, p: f64) -> Self {
        let Self((_, b, c, d)) = self;
        Self((p, b, c, d))
    }

    /// The probability of removing an existing connection between neurons.  Represented as the relative size of this number to the others.
    pub const fn remove_connection(self, p: f64) -> Self {
        let Self((a, _, c, d)) = self;
        Self((a, p, c, d))
    }

    /// The probability of adding a new neuron.  Represented as the relative size of this number to the others.
    pub const fn add_neuron(self, p: f64) -> Self {
        let Self((a, b, _, d)) = self;
        Self((a, b, p, d))
    }

    /// The probability of adding a new bias neuron. Represented as the relative size of this number to the others.
    pub const fn add_bias(self, p: f64) -> Self {
        let Self((a, b, c, _)) = self;
        Self((a, b, c, p))
    }
}
