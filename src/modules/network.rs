use modules::functions::*;

#[derive(Clone)]
pub struct Gene {
    // type 0 = neuron
    // type 1 = input
    // type 2 = Jumper Forward
    // type 3 = Jumper Recurrent
    pub value: i32,
    pub typ: i32,
    pub weight: f64,
    pub inputs: i32,
    pub id: usize
}

pub struct Network {
    pub genome: Vec<Gene>
}

#[derive(Debug)]
pub struct Stack {
    pub vec: Vec<i32>
}

impl Gene {
    pub fn new(value: i32, typ: i32, weight: f64, inputs: i32, id: usize) -> Gene {
        Gene {
            value: value,
            typ: typ,
            weight: weight,
            inputs: inputs,
            id: id
        }
    }
}

impl Network {
    pub fn evaluate(&mut self) -> Vec<f64> {
        // evaluate back to front
        // push inputs onto stack
        // pop values from stack when a neuron is encountered
        // push neuron output onto stack
        let genome = reverse(&self.genome);
        let mut stack = Stack::new();

        for i in 0..genome.len() as usize {
            let gene = &genome[i];
            match gene.typ {
                0 => {
                    let vec = stack.pop(gene.inputs);
                    self.genome[i].value = sum_vec(&vec);
                    stack.push(self.genome.value.clone());
                },
                1 => stack.push(gene.value.clone()),
                _ => unreachable!()
            }
        }

        vec![0.0]
    }

    pub fn new() -> Network {
        Network {
            genome: Vec::new()
        }
    }
}

impl Stack {
    pub fn pop(&mut self, count: i32) -> Vec<i32> {
        let mut result = Vec::new();
        for _ in 0..count {
            let item = self.vec.pop();
            result.push(match item {
                Some(x) => x,
                None => panic!("pop failed")
            })
        }
        result
    }

    pub fn push(&mut self, value: i32) {
        self.vec.push(value);
    }

    pub fn new() -> Stack {
        Stack {
            vec: Vec::new()
        }
    }
}
