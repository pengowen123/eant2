pub trait Mutation {
    fn add_subnetwork(&mut self, input: usize, output: usize);
    fn add_connection(&mut self, input: usize, output: usize);
    fn add_bias(&mut self, output: usize);
    fn remove_connection(&mut self, input: usize, output: usize);
}
