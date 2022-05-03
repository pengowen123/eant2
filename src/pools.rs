// use cge::gene::Gene;
// /! # Object Pools
// /! These object pools function to eliminate allocation costs over time,
// /! by allowing the typed reuse of previously allocated memory.
// /! 
// /! Another benefit of this approach is that when a heap allocation is reused, it is reused on a most-recently-released basis.
// /! This naturally increases CPU cache efficiency.

// use static_init::dynamic;
// use object_pool::Pool;

// / Genome object pool. Initially contains 8 empty genomes. Will grow on demand.
// #[dynamic] pub static GENOMES: Pool<Vec<Gene>> = Pool::new(8, Vec::with_capacity(0));