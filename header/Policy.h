#pragma once

#include "RandomGenerator.h"
#include "Simulator.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

class Policy {
public:
    // Initializes with:
    // - max_timestep: runs t = 0..max_timestep inclusive
    // - epsilon: exploration probability in [0.0, 1.0] (used when ucb_c == 0.0)
    // - initial_value: Q_0 for optimistic initial estimates (default 0.0)
    // - ucb_c: UCB exploration degree c >= 0.0 (0.0 disables UCB)
    // - seed: optional RNG seed
    Policy(size_t max_timestep,
           double epsilon = 0.1,
           double initial_value = 0.0,
           double ucb_c = 0.0,
           std::optional<uint64_t> seed = std::nullopt);

    // Executes the policy against the simulator from t = 0 to max_timestep.
    // Returns a vector of observed rewards of size (max_timestep + 1).
    std::vector<double> run(Simulator& simulator);

private:
    size_t max_timestep_;
    double epsilon_;
    double initial_value_;
    double ucb_c_;
    RandomGenerator rng_;
};