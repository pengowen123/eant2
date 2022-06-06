//! Utilities for working with CGE networks

use cge::network::{MismatchedLengthsError, NotEnoughInputsError};

use std::ops::Deref;

/// The initial value for the weight of any new gene in a network.
pub const INITIAL_WEIGHT_VALUE: f64 = 1.0;

/// A CGE network.
pub type Network = cge::Network<f64>;

/// A view into a [`Network`] that only provides restricted set of operations.
pub struct NetworkView<'a>(&'a mut Network);

impl<'a> NetworkView<'a> {
    pub fn new(network: &'a mut Network) -> Self {
        Self(network)
    }

    /// See [`Network::evaluate`][Network::evaluate].
    pub fn evaluate(&mut self, inputs: &[f64]) -> Result<&[f64], NotEnoughInputsError> {
        self.0.evaluate(inputs)
    }

    /// See [`Network::clear_state`][Network::clear_state].
    pub fn clear_state(&mut self) {
        self.0.clear_state();
    }

    /// See [`Network::recurrent_state_len`][Network::recurrent_state_len].
    pub fn recurrent_state_len(&mut self) -> usize {
        self.0.recurrent_state_len()
    }

    /// See [`Network::recurrent_state`][Network::recurrent_state].
    pub fn recurrent_state(&mut self) -> impl Iterator<Item = f64> + '_ {
        self.0.recurrent_state()
    }

    /// See [`Network::set_recurrent_state`][Network::set_recurrent_state].
    pub fn set_recurrent_state(&mut self, state: &[f64]) -> Result<(), MismatchedLengthsError> {
        self.0.set_recurrent_state(state)
    }

    /// See [`Network::map_recurrent_state`][Network::map_recurrent_state].
    pub fn map_recurrent_state<F: FnMut(usize, &mut f64)>(&mut self, f: F) {
        self.0.map_recurrent_state(f)
    }
}

// Allow access to immutable methods from `Network`
impl<'a> Deref for NetworkView<'a> {
    type Target = Network;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
