# eant2

[![Build Status](https://travis-ci.org/pengowen123/eant2.svg?branch=master)](https://travis-ci.org/pengowen123/eant2)

This library is currently broken. I am looking into the problem, which seems to be caused by the CMA-ES crate. Once it is functional it will be released.

An implementation of the [EANT2](http://www.informatik.uni-kiel.de/inf/Sommer/doc/Publications/nts/SiebelSommer-IJHIS2007.pdf) algorithm in Rust. EANT2 is an evolutionary neural network training algorithm. It solves many of the problems associated with evolving topology of neural networks, as well as introducing new concepts. There are already libraries for neural network training, but require a lot of setup to make use of. This library aims to change that. A small number of (completely optional) parameters makes EANT2 easy to use. The goal of this library is to make it easy to include neural networks in any project.

This library has two other parts: [CMA-ES](https://github.com/pengowen123/cmaes) and [CGE][1].

## Usage

Add this to your Cargo.toml:

```
[dependencies]

eant2 = { git = "https://github.com/pengowen123/eant2" }
```

And this to your crate root:

```rust
extern crate eant2;
```

## Example
First add all the things we will use later
```rust
extern crate gym_rs;
extern crate eant2;
extern crate cge;

use gym_rs::{CartPoleEnv, GymEnv, ActionType, Viewer};
use cge::{Activation};
use eant2::{EANT2Options, eant_loop, Network, NNFitnessFunction};
```

Then, define your own environment in which to train the neural network. This example uses the 
[gym-rs](https://www.github.com/MathisWellmann/gym-rs) crate for a cart pole environment. 
It is a well established benchmark environment in academia and is a translation from OpenAI's gym implementation in Python.

```rust
#[derive(Clone)]
pub struct MyEnv{}

impl MyEnv {
    fn fitness(&self, net: &mut Network) -> f64 {
        let mut env = CartPoleEnv::default();
        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_reward: f64 = 0.0;
        while !end {
            if total_reward > 200.0 {
                break
            }
            let output = net.evaluate(&state);

            let action: ActionType = if output[0] < -0.0 {
                ActionType::Discrete(0)
            } else {
                ActionType::Discrete(1)
            };
            let (s, reward, done, _info) = env.step(action);
            end = done;
            total_reward += reward;
            state = s;
        }

        // println!("total_reward: {}", total_reward);
        total_reward
    }
}

impl NNFitnessFunction for MyEnv {
    fn get_fitness(&self, net: &mut Network) -> f64 {
        // Get the avg score of 10 sample runs
        (0..10).collect::<Vec<usize>>().iter().map(|_| self.fitness(net)).sum::<f64>() / 10.0
    }
}
```

Then set up rendering for the final champion.

```rust
fn render_champion(net: &mut Network) {
    let mut env = CartPoleEnv::default();
    let mut state: Vec<f64> = env.reset();

    let mut viewer = Viewer::new(1080, 1080);

    let mut end: bool = false;
    let mut total_reward: f64 = 0.0;
    while !end {
        if total_reward > 300.0 {
            println!("win!!!");
            break;
        }
        let output = net.evaluate(&state);

        let action: ActionType = if output[0] < 0.0 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };
        let (s, reward, done, _info) = env.step(action);
        end = done;
        total_reward += reward;
        state = s;

        env.render(&mut viewer);
    }
}
```

Finally, run the optimization for 100 generations and a Tanh neural network activation function.
```rust

fn main() {
    let options = EANT2Options::new(4, 1)
        .print(true)
        .max_generations(100)
        .transfer_function(Activation::Tanh);
    let mut solution = eant_loop(&MyEnv{}, options);

    println!("solution: {:?}", solution);

    render_champion(&mut solution.0);

    // save champion to file using cge crate
    solution.0.save_to_file("./examples/gym_cart_pole_champion.cge").unwrap();
}
```

See example folder for the whole file.

Run using: 

```
cargo run --release --example gym_cart_pole --features="gym-rs"
```

As you will see, this crate currently fails to solve the cart pole environment.
Feel free to be a hero and try to fix it.

To use it elsewhere, use the [CGE library][1]. With it, you can load the network, and evaluate and reset it as many times as you wish. For a freestanding version (for embedded systems), use [cge-core][2].

See the [documentation][3] for complete instructions as well as tips for improving performance of the algorithm. For docmentation on the usage of the neural networks, see the documentation for the CGE library [here](https://pengowen123.github.io/cge/cge/index.html).

## Contributing

If you encounter a bug, please open an issue and I will try to get it fixed. If you have a suggestion, feel free to open an issue or implement it yourself and open a pull request.

[1]: https://github.com/pengowen123/cge
[2]: https://github.com/pengowen123/cge-core
[3]: https://pengowen123.github.io/eant2/eant2/index.html