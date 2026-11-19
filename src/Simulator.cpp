#include "..\header\Simulator.h"
#include <algorithm>
#include <stdexcept>

// Mode 1: Simulator decides everything
Simulator::Simulator(std::optional<uint64_t> seed)
    : rng_(seed.has_value() ? RandomGenerator(*seed) : RandomGenerator()) {
    num_actions_ = static_cast<size_t>(rng_.uniform_int(1, 1000));
    generate_random_parameters();
}

// Mode 2: User provides num_actions, Simulator decides means and stddevs
Simulator::Simulator(size_t num_actions, std::optional<uint64_t> seed)
    : num_actions_(num_actions),
      rng_(seed.has_value() ? RandomGenerator(*seed) : RandomGenerator()) {
    if (num_actions_ == 0) {
        throw std::invalid_argument("Simulator: num_actions must be greater than 0");
    }
    generate_random_parameters();
}

// Mode 3: User provides num_actions, means, and stddevs
Simulator::Simulator(size_t num_actions,
                     const std::vector<double>& means,
                     const std::vector<double>& stddevs,
                     std::optional<uint64_t> seed)
    : means_(means),
      stddevs_(stddevs),
      num_actions_(num_actions),
      rng_(seed.has_value() ? RandomGenerator(*seed) : RandomGenerator()) {
    if (num_actions_ == 0) {
        throw std::invalid_argument("Simulator: num_actions must be greater than 0");
    }
    if (means_.size() != num_actions_ || stddevs_.size() != num_actions_) {
        throw std::invalid_argument("Simulator: means and stddevs sizes must equal num_actions");
    }
    for (double sd : stddevs_) {
        if (sd < 0.0) {
            throw std::invalid_argument("Simulator: standard deviations must be >= 0.0");
        }
    }
}

void Simulator::generate_random_parameters() {
    means_.resize(num_actions_);
    stddevs_.resize(num_actions_);

    for (size_t i = 0; i < num_actions_; ++i) {
        means_[i] = rng_.uniform_real(-1.0, 1.0);
        stddevs_[i] = rng_.uniform_real(1.0, 2.0);
    }
}

double Simulator::get_current_value_of_action(size_t action) {
    if (action >= num_actions_) {
        throw std::out_of_range("Simulator::get_current_value_of_action: action index out of range");
    }
    return rng_.normal(means_[action], stddevs_[action]);
}

double Simulator::optimal_total_reward() const {
    const double max_mean = *std::max_element(means_.begin(), means_.end());
    return max_mean * static_cast<double>(num_actions_);
}