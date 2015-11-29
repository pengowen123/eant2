extern crate eant_rust;

use eant_rust::modules::network;

fn main() {
	let bar = network::Stack {
		vec: vec![0, 1, 2]
	};
    println!("{:?}", bar);
}
