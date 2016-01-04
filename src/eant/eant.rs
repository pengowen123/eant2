// This is the main algorithm, which mutates neural networks, then uses CMA-ES to optimize them
// High fitness networks are kept to be mutated, while low fitness networks are discarded
// The loop ends when a solution is found

use cge::network::Network;
use cmaes::cmaes::cmaes_loop;

use eant::fitness;
struct Foo;
impl fitness::FitnessFunction for Foo {
    fn get_fitness(network: &mut Network) -> f64 {
        network.step(vec![0.0], false)[0]
    }
}

pub fn eant_loop(threads: u8) {
    let networks = vec![Network::new()];
    cmaes_loop(Foo, &networks, 4, threads);
}
