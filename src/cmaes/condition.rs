pub enum CMAESEndConditions {
    Stabilized(f64, usize),
    FitnessThreshold(f64),
    MaxGenerations(usize)
}
