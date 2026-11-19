#include "Policy.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

Policy::Policy(size_t max_timestep,
               double epsilon,
               double initial_value,
               double ucb_c,
               std::optional<uint64_t> seed)
    : max_timestep_(max_timestep),
      epsilon_(epsilon),
      initial_value_(initial_value),
      ucb_c_(ucb_c),
      rng_(seed.has_value() ? RandomGenerator(*seed) : RandomGenerator()) {
    if (epsilon_ < 0.0 || epsilon_ > 1.0) {
        throw std::invalid_argument("Policy: epsilon must be in [0.0, 1.0]");
    }
    if (ucb_c_ < 0.0) {
        throw std::invalid_argument("Policy: ucb_c must be >= 0.0");
    }
}

std::vector<double> Policy::run(Simulator& simulator) {
    const size_t num_actions = simulator.num_actions();
    if (num_actions == 0) {
        throw std::invalid_argument("Policy::run: simulator must have at least 1 action");
    }

    std::vector<double> estimated_values(num_actions, initial_value_);
    std::vector<size_t> action_counts(num_actions, 0);

    std::vector<double> rewards;
    rewards.reserve(max_timestep_ + 1);

    for (size_t t = 0; t <= max_timestep_; ++t) {
        size_t chosen_action = 0;

        if (ucb_c_ > 0.0) {
            // UCB selection: argmax_a [ Q_t(a) + c * sqrt(ln(T) / N_t(a)) ]
            // Using T = t + 1 (1-based step count so T >= 1 and ln(T) >= 0)
            const double log_T = std::log(static_cast<double>(t + 1));
            std::vector<double> ucb_scores(num_actions, 0.0);

            for (size_t a = 0; a < num_actions; ++a) {
                if (action_counts[a] == 0) {
                    // Unselected actions have infinite upper confidence bound
                    ucb_scores[a] = std::numeric_limits<double>::infinity();
                } else {
                    const double bonus = ucb_c_ * std::sqrt(log_T / static_cast<double>(action_counts[a]));
                    ucb_scores[a] = estimated_values[a] + bonus;
                }
            }

            const double max_score = *std::max_element(ucb_scores.begin(), ucb_scores.end());
            std::vector<size_t> best_actions;
            for (size_t a = 0; a < num_actions; ++a) {
                if (ucb_scores[a] == max_score) {
                    best_actions.push_back(a);
                }
            }

            const size_t tie_idx = static_cast<size_t>(
                rng_.uniform_int(0, static_cast<int64_t>(best_actions.size() - 1))
            );
            chosen_action = best_actions[tie_idx];
        } else if (rng_.bernoulli(epsilon_)) {
            // Explore uniformly at random
            chosen_action = static_cast<size_t>(
                rng_.uniform_int(0, static_cast<int64_t>(num_actions - 1))
            );
        } else {
            // Exploit greedy action with random tie-breaking
            const double max_avg = *std::max_element(estimated_values.begin(), estimated_values.end());
            std::vector<size_t> best_actions;
            for (size_t a = 0; a < num_actions; ++a) {
                if (estimated_values[a] == max_avg) {
                    best_actions.push_back(a);
                }
            }
            const size_t tie_idx = static_cast<size_t>(
                rng_.uniform_int(0, static_cast<int64_t>(best_actions.size() - 1))
            );
            chosen_action = best_actions[tie_idx];
        }

        const double reward = simulator.get_current_value_of_action(chosen_action);
        rewards.push_back(reward);

        // Incremental sample-average update: Q_n = Q_{n-1} + (R_n - Q_{n-1}) / n
        action_counts[chosen_action]++;
        const double step_size = 1.0 / static_cast<double>(action_counts[chosen_action]);
        estimated_values[chosen_action] += step_size * (reward - estimated_values[chosen_action]);
    }

    return rewards;
}