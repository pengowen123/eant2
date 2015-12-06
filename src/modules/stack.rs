// This might be deleted
pub struct Stack {
    pub vec: Vec<i32>
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
