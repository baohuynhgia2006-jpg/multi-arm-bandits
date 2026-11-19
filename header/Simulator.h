#pragma once

#include "RandomGenerator.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

class Simulator {
public:
    // Mode 1: Simulator randomly picks num_actions in [1, 1000],
    // means in [-1.0, 1.0], and stddevs in [1.0, 2.0].
    explicit Simulator(std::optional<uint64_t> seed = std::nullopt);

    // Mode 2: User specifies num_actions; Simulator randomly picks
    // means in [-1.0, 1.0] and stddevs in [1.0, 2.0] for each action.
    explicit Simulator(size_t num_actions, std::optional<uint64_t> seed = std::nullopt);

    // Mode 3: User specifies num_actions, means, and stddevs explicitly.
    Simulator(size_t num_actions,
              const std::vector<double>& means,
              const std::vector<double>& stddevs,
              std::optional<uint64_t> seed = std::nullopt);

    // Samples a reward from N(means[action], stddevs[action])
    double get_current_value_of_action(size_t action);

    // Returns max(means) * num_actions
    double optimal_total_reward() const;

    // Inside Simulator public section in simulator.hpp:
    size_t num_actions() const noexcept { return num_actions_; }

private:
    void generate_random_parameters();

    std::vector<double> means_;
    std::vector<double> stddevs_;
    size_t num_actions_;
    RandomGenerator rng_;
};