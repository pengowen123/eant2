// This is the main algorithm, which mutates neural networks, then uses CMA-ES to optimize them
// High fitness networks are kept to be mutated, while low fitness networks are discarded
// The loop ends when a solution is found

use eant::network;
use cmaes::cmaes;
