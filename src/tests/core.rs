// NOTE: Modify this code to complete the todo in lib.rs


use crate::{*};

use crate::FitnessFunction;

#[derive(Clone)]
struct Foo;
impl FitnessFunction for Foo {
  fn get_fitness(&self, network: &mut Network) -> f64 {
    let data = [
      ([0.0, 0.0], 0.0),
      ([1.0, 0.0], 1.0),
      ([0.0, 1.0], 1.0),
      ([1.0, 1.0], 0.0)
    ];

    let mut fitness = 0.0;

    for set in data.iter() {
      let result = network.evaluate(&set.0)[0];
      fitness += (result - set.1).abs();
    }

    fitness
  }
}

#[cfg(test)]
mod test {
  use crate::{eant2::EANT2, Activation};
  use super::Foo;

  #[test]
  fn main() {
    let eant = 
      EANT2::builder()
        .inputs(2)
        .outputs(1)
        .activation(Activation::Threshold)
        .print()
        .build();

      let (network, fitness) = eant.run(&Foo);

      println!("{:?}", fitness);
      println!("{:?}", network);
  }
}