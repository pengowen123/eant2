// A simple stack with push and pop methods used for evaluating a network

pub struct Stack {
    pub vec: Vec<f64>,
}

impl Stack {
    pub fn pop(&mut self, count: i32) -> Vec<f64> {
        let mut result = Vec::new();
        for _ in 0..count {
            let item = self.vec.pop();
            result.push(match item {
                Some(x) => x,
                None => panic!("pop failed"),
            });
        }
        result
    }

    pub fn push(&mut self, value: f64) {
        self.vec.push(value);
    }

    pub fn new() -> Stack {
        Stack { vec: Vec::new() }
    }
}
