use std::convert::TryFrom;
use rand::{ThreadRng, distributions::{Range, IndependentSample}};
use crate::mutation::MutationType;

/// Describes the relative probabilites of different structural mutations.
/// 
/// Use `MutationProbabilitiesFloat` for a more ergonomic API that allows for entry as f64s.
/// 
/// Note: fits in a single 64bit register. Copy, don't reference.
#[derive(Copy, Clone)]
pub struct MutationProbabilities(u64);

impl From<(u16, u16, u16, u16)> for MutationProbabilities {
  fn from(probabilities: (u16, u16, u16, u16)) -> Self {
    MutationProbabilities::assemble(probabilities)
  }
}

impl MutationProbabilities {
  pub const fn assemble(probabilities: (u16, u16, u16, u16)) -> Self {
    MutationProbabilities({
      (probabilities.0 as u64)       |
      (probabilities.1 as u64) >> 16 |
      (probabilities.2 as u64) >> 32 |
      (probabilities.3 as u64) >> 48
    })
  }

  /// The probability of adding a new connection between neurons.  Represented as the relative size of this `u16` to the others.
  pub fn add_connection(self, probability: u16) -> Self {
    // clear the 0th slot
    let cleared = self.0 & (!0xFFFF_0000_0000_0000);
    MutationProbabilities(cleared | (probability as u64))
  }

  /// The probability of removing an existing connection between neurons.  Represented as the relative size of this `u16` to the others.
  pub fn remove_connection(self, probability: u16) -> Self {
    // clear the 1st slot
    let cleared = self.0 & (!0x0000_FFFF_0000_0000);
    MutationProbabilities(cleared | (probability as u64))
  }

  /// The probability of adding a new neuron.  Represented as the relative size of this `u16` to the others.
  pub fn add_neuron(self, probability: u16) -> Self {
    // clear the 2nd slot
    let cleared = self.0 & (!0x0000_0000_FFFF_0000);
    MutationProbabilities(cleared | (probability as u64))
  }

  /// The probability of adding a new bias neuron.  Represented as the relative size of this `u16` to the others.
  pub fn add_bias(self, probability: u16) -> Self {
    // clear the 3rd slot
    let cleared = self.0 & (!0x0000_0000_0000_FFFF);
    MutationProbabilities(cleared | (probability as u64))
  }

  /// Generate a random mutation according to the probabilities contained in this struct.
  pub(crate) fn generate(&self, rng: &mut ThreadRng) -> MutationType {
    let packed = self.0;

    let a = ( packed & 0xFFFF_0000_0000_0000       ) as u32;
    let b = ((packed & 0x0000_FFFF_0000_0000) << 16) as u32;
    let c = ((packed & 0x0000_0000_FFFF_0000) << 32) as u32;
    let d = ((packed & 0x0000_0000_0000_FFFF) << 48) as u32;

    let sum = a + b + c + d;

    let n = Range::new(0, sum).ind_sample(rng);
    
    let mut sum = 0;
    if n >= sum && n < sum + b { return MutationType::AddConnection; }    sum += a;
    if n >= sum && n < sum + c { return MutationType::RemoveConnection; } sum += b;
    if n >= sum && n < sum + d { return MutationType::AddNode; }
    else {                       return MutationType::AddBias; }
  }
}

/// Easier to use mutation probabilities.
/// - Input is a tuple of probabilities for each mutation.
/// - Automatically normalized upon conversion to `MutationProbabilities`.
#[derive(Copy, Clone)]
pub struct MutationProbabilitiesFloat((f64, f64, f64, f64));

/// If you receive this, it means you've committed a probability crime.
/// - All probability entries must be non-negative: `[0, +inf)`.
/// - All probability entries must not sum to zero (there must be at least one non-zero entry).
pub struct IllSpecifiedProbabilityError;

impl TryFrom<MutationProbabilitiesFloat> for MutationProbabilities {
  type Error = IllSpecifiedProbabilityError;

  fn try_from(p: MutationProbabilitiesFloat) -> Result<Self, IllSpecifiedProbabilityError> {
    let p = p.0;
    let sum = p.0 + p.1 + p.2 + p.3;

    // all probabilities must sum above 0.0,
    // all probabilities must be non-negative.
    if sum == 0.0 || p.0 < 0. || p.1 < 0. || p.2 < 0. || p.3 < 0. { return Err(IllSpecifiedProbabilityError); }

    Ok(MutationProbabilities::from((
      ((u16::MAX as f64) * p.0 / sum) as u16,
      ((u16::MAX as f64) * p.1 / sum) as u16,
      ((u16::MAX as f64) * p.2 / sum) as u16,
      ((u16::MAX as f64) * p.3 / sum) as u16
    )))
  }
}

impl MutationProbabilitiesFloat {
  /// All probabilities start as zero.
  /// - You must provide at least one non-zero probability.
  /// - No probability may be negative.
  pub fn zeros() -> Self { MutationProbabilitiesFloat((0., 0., 0., 0.)) }
}

impl MutationProbabilitiesFloat {
  /// The probability of adding a new connection between neurons.  Represented as the relative size of this `u16` to the others.
  pub fn add_connection(mut self, probability: f64) -> Self { self.0.0 = probability; self }

  /// The probability of removing an existing connection between neurons.  Represented as the relative size of this `u16` to the others.
  pub fn remove_connection(mut self, probability: f64) -> Self { self.0.1 = probability; self }

  /// The probability of adding a new neuron.  Represented as the relative size of this `u16` to the others.
  pub fn add_neuron(mut self, probability: f64) -> Self { self.0.2 = probability; self }

  /// The probability of adding a new bias neuron.  Represented as the relative size of this `u16` to the others.
  pub fn add_bias(mut self, probability: f64) -> Self { self.0.3 = probability; self }
}
