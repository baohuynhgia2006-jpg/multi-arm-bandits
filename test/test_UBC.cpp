#include "Policy.h"
#include "Simulator.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct ExperimentMetrics {
    std::string label;
    std::vector<double> avg_cumulative_reward;
    std::vector<double> avg_theoretical_optimal;
    std::vector<double> optimal_percentage;
    double max_optimal_pct = -1e9;
    size_t best_timestep = 0;
};

ExperimentMetrics run_experiment(const std::string& label,
                                   size_t num_runs,
                                   size_t num_actions,
                                   size_t max_timestep,
                                   double epsilon,
                                   double initial_value,
                                   double ucb_c,
                                   uint64_t base_seed) {
    const size_t total_steps = max_timestep + 1;
    std::vector<double> sum_cumulative_reward(total_steps, 0.0);
    std::vector<double> sum_theoretical_optimal(total_steps, 0.0);
    std::vector<double> sum_optimal_ratio(total_steps, 0.0);

    for (size_t run = 0; run < num_runs; ++run) {
        const uint64_t sim_seed = base_seed + run * 2;
        const uint64_t pol_seed = base_seed + run * 2 + 1;

        Simulator sim(num_actions, sim_seed);
        Policy policy(max_timestep, epsilon, initial_value, ucb_c, pol_seed);

        const std::vector<double> step_rewards = policy.run(sim);
        const double max_mean = sim.optimal_total_reward() / static_cast<double>(num_actions);

        double cumulative_reward = 0.0;
        for (size_t t = 0; t < total_steps; ++t) {
            cumulative_reward += step_rewards[t];
            const double optimal_cumulative = max_mean * static_cast<double>(t + 1);

            sum_cumulative_reward[t] += cumulative_reward;
            sum_theoretical_optimal[t] += optimal_cumulative;

            if (optimal_cumulative != 0.0) {
                sum_optimal_ratio[t] += (cumulative_reward / optimal_cumulative) * 100.0;
            }
        }
    }

    ExperimentMetrics metrics;
    metrics.label = label;
    metrics.avg_cumulative_reward.resize(total_steps);
    metrics.avg_theoretical_optimal.resize(total_steps);
    metrics.optimal_percentage.resize(total_steps);

    const double inv_runs = 1.0 / static_cast<double>(num_runs);
    for (size_t t = 0; t < total_steps; ++t) {
        metrics.avg_cumulative_reward[t] = sum_cumulative_reward[t] * inv_runs;
        metrics.avg_theoretical_optimal[t] = sum_theoretical_optimal[t] * inv_runs;
        metrics.optimal_percentage[t] = sum_optimal_ratio[t] * inv_runs;

        if (metrics.optimal_percentage[t] > metrics.max_optimal_pct) {
            metrics.max_optimal_pct = metrics.optimal_percentage[t];
            metrics.best_timestep = t;
        }
    }

    return metrics;
}

int main() {
    constexpr size_t NUM_RUNS = 2000;
    constexpr size_t NUM_ACTIONS = 10;
    constexpr size_t MAX_TIMESTEP = 999;

    std::cout << "Running " << NUM_RUNS << " policies per configuration ("
              << NUM_ACTIONS << " actions, " << (MAX_TIMESTEP + 1) << " steps)...\n\n";

    const std::vector<ExperimentMetrics> experiments = {
        run_experiment("Eps-Greedy (e=0.10)", NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.10, 0.0, 0.0, 100000ULL),
        run_experiment("UCB (c=0.50)",        NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.00, 0.0, 0.5, 100000ULL),
        run_experiment("UCB (c=1.00)",        NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.00, 0.0, 1.0, 100000ULL),
        run_experiment("UCB (c=2.00)",        NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.00, 0.0, 2.0, 100000ULL)
    };

    // Table Header
    std::cout << std::left
              << std::setw(6)  << "Step"
              << " | " << std::setw(15) << "Theoretical Opt";
    for (const auto& exp : experiments) {
        std::cout << " | " << std::setw(24) << exp.label;
    }
    std::cout << "\n" << std::string(130, '-') << "\n";

    std::cout << std::fixed << std::setprecision(2);

    for (size_t t = 0; t <= MAX_TIMESTEP; ++t) {
        std::cout << std::left << std::setw(6) << t
                  << " | " << std::right << std::setw(15) << experiments[0].avg_theoretical_optimal[t];
        for (const auto& exp : experiments) {
            std::cout << " | " << std::setw(10) << exp.avg_cumulative_reward[t]
                      << " / " << std::setw(6) << exp.optimal_percentage[t] << "%";
        }
        std::cout << "\n";
    }

    // Peak Optimal Percentage Summary
    std::cout << std::string(130, '=') << "\n"
              << "Peak Optimal Performance:\n";
    for (const auto& exp : experiments) {
        std::cout << "  " << std::left << std::setw(24) << exp.label
                  << ": " << std::right << std::setw(6) << exp.max_optimal_pct
                  << "% at timestep " << exp.best_timestep << "\n";
    }

    return 0;
}