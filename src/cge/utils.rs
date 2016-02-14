pub struct Stack {
    pub data: Vec<f64>,
}

impl Stack {
    pub fn new() -> Stack {
        Stack { data: Vec::new() }
    }

    pub fn pop(&mut self, count: usize) -> Option<Vec<f64>> {
        if count >= self.data.len() {
            return None;
        }

        let mut result = Vec::new();

        for _ in 0..count {
            result.push(self.data.pop().unwrap());
        }

        Some(result)
    }

    pub fn push(&mut self, value: f64) {
        self.data.push(value);
    }
}
