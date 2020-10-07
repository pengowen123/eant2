extern crate gym_rs;
extern crate eant2;
extern crate cge;

use gym_rs::{CartPoleEnv, GymEnv, ActionType, Viewer};
use cge::{Activation};
use eant2::{EANT2Options, eant_loop, Network, NNFitnessFunction};

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

fn main() {
    let options = EANT2Options::new(4, 1)
        .print(true)
        .max_generations(10)
        .transfer_function(Activation::Tanh);
    let mut solution = eant_loop(&MyEnv{}, options);

    println!("solution: {:?}", solution);

    render_champion(&mut solution.0);

    // save champion to file using cge crate
    solution.0.save_to_file("./examples/gym_cart_pole_champion.cge").unwrap();
}