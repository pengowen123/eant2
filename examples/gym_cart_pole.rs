extern crate cge;
extern crate eant2;
extern crate gym_rs;

use cge::encoding::{Metadata, WithRecurrentState};
use eant2::eant2::EANT2;
use eant2::{Activation, FitnessFunction, Network, NetworkView};
use gym_rs::{ActionType, CartPoleEnv, GifRender, GymEnv};

#[derive(Clone)]
pub struct MyEnv;

impl MyEnv {
    fn fitness(&self, net: &mut NetworkView) -> f64 {
        let mut env = CartPoleEnv::default();
        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_reward: f64 = 0.0;
        while !end {
            if total_reward > 1000.0 {
                break;
            }
            let output = net.evaluate(&state).unwrap();

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

        -total_reward
    }
}

impl FitnessFunction for MyEnv {
    fn fitness(&self, mut net: NetworkView) -> f64 {
        // Get the worst score of 10 sample runs
        (0..10)
            .collect::<Vec<usize>>()
            .iter()
            .map(|_| self.fitness(&mut net))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

fn render_champion(net: &mut Network) {
    println!("Rendering solution");
    let mut env = CartPoleEnv::default();
    let mut state: Vec<f64> = env.reset();

    let mut render = GifRender::new(540, 540, "./examples/cart_pole_champion.gif", 20).unwrap();

    let mut end: bool = false;
    let mut total_reward: f64 = 0.0;
    while !end {
        if total_reward > 300.0 {
            println!("win!!!");
            break;
        }
        let output = net.evaluate(&state).unwrap();

        let action: ActionType = if output[0] < 0.0 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };
        let (s, reward, done, _info) = env.step(action);
        end = done;
        total_reward += reward;
        state = s;

        env.render(&mut render);
    }
}

fn main() {
    let mut eant2 = EANT2::builder()
        .inputs(4)
        .outputs(1)
        .print()
        .activation(Activation::Tanh)
        .build();
    let (mut network, _) = eant2.run(&MyEnv);

    render_champion(&mut network);

    println!("Saving solution to file");
    // save champion to file using cge crate
    network
        .to_file(
            Metadata::new(Some(
                "A solution network for the gym cart pole problem.".into(),
            )),
            (),
            WithRecurrentState(false),
            "./examples/gym_cart_pole_champion.cge",
            true,
        )
        .unwrap();
}
