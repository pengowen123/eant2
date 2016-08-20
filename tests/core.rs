// NOTE: Modify this code to complete the todo in lib.rs

//extern crate eant2;

//extern crate cmaes;
//use cmaes::{cmaes_loop, FitnessFunction, CMAESOptions};

//use eant2::*;

//#[derive(Clone)]
//struct Foo;

//impl NNFitnessFunction for Foo {
    //fn get_fitness(&self, network: &mut Network) -> f64 {
        //let data = [
            //([0.0, 0.0], 0.0),
            //([1.0, 0.0], 1.0),
            //([0.0, 1.0], 1.0),
            //([1.0, 1.0], 0.0)
        //];

        //let mut fitness = 0.0;

        //for set in data.iter() { 
            //let result = network.evaluate(&set.0)[0];
            //fitness += (result - set.1).abs();
        //}

        //fitness
    //}
//}

//impl FitnessFunction for Foo {
    //fn get_fitness(&self, parameters: &[f64]) -> f64 {
        //let mut network = Network::from_str("1:
                                 //n 0 0 2,
                                    
                                    //n 0 2 3,
                                        //i 0 0,
                                        //i 0 1,
                                        //b 0,
                                    
                                    //n 0 3 2,
                                        //i 0 0,
                                        //i 0 1
                                //").expect("test");

        //network.genome.iter_mut().enumerate().map(|(i, x)| x.weight = parameters[i]);

        //let data = [
            //([0.0, 0.0], 0.0),
            //([1.0, 0.0], 1.0),
            //([0.0, 1.0], 1.0),
            //([1.0, 1.0], 0.0)
        //];

        //let mut fitness = 0.0;

        //for set in data.iter() { 
            //let result = network.evaluate(&set.0)[0];
            //fitness += (result - set.1).abs();
        //}

        //fitness
    //}
//}

//fn main() {
    //let options = EANT2Options::new(2, 1)
        //.print(true)
        //.fitness_threshold(0.01)
        //.transfer_function(TransferFunction::Threshold);

    //let mut solution = eant_loop(&Foo, options).0;
    
    //println!("{}", Foo.get_fitness(&mut solution));
    //foo(Foo);

    //let test = Network::from_str("1:
                                 //n 0 0 2,
                                    
                                    //n 0 2 3,
                                        //i 0 0,
                                        //i 0 1,
                                        //b 0,
                                    
                                    //n 0 3 2,
                                        //i 0 0,
                                        //i 0 1
                                //").expect("test");

    //let options = CMAESOptions::default(8);

    //let solution = cmaes_loop(&Foo, options);
    //println!("{:?}", solution);
//}
