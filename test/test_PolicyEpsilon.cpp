#include "Policy.h"
#include "Simulator.h"
#include <iomanip>
#include <iostream>
#include <vector>

struct ExperimentMetrics {
    std::vector<double> avg_cumulative_reward;
    std::vector<double> avg_theoretical_optimal;
    std::vector<double> optimal_percentage;
    double max_optimal_pct = -1e9;
    size_t best_timestep = 0;
};

ExperimentMetrics run_experiment(size_t num_runs,
                                   size_t num_actions,
                                   size_t max_timestep,
                                   double epsilon,
                                   uint64_t base_seed) {
    const size_t total_steps = max_timestep + 1;
    std::vector<double> sum_cumulative_reward(total_steps, 0.0);
    std::vector<double> sum_theoretical_optimal(total_steps, 0.0);
    std::vector<double> sum_optimal_ratio(total_steps, 0.0);

    for (size_t run = 0; run < num_runs; ++run) {
        const uint64_t sim_seed = base_seed + run * 2;
        const uint64_t pol_seed = base_seed + run * 2 + 1;

        Simulator sim(num_actions, sim_seed);
        Policy policy(max_timestep, epsilon, pol_seed);

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
    constexpr size_t NUM_ACTIONS = 10;      // Classic 10-armed testbed
    constexpr size_t MAX_TIMESTEP = 999;    // Runs t = 0..999 (1000 steps total)

    std::cout << "Running " << NUM_RUNS << " policies per epsilon ("
              << NUM_ACTIONS << " actions, " << (MAX_TIMESTEP + 1) << " steps)...\n\n";

    // Identical base seed evaluates all four epsilon classes against the same 2000 environments
    const ExperimentMetrics eps_000 = run_experiment(NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.00, 100000ULL);
    const ExperimentMetrics eps_001 = run_experiment(NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.01, 100000ULL);
    const ExperimentMetrics eps_010 = run_experiment(NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.10, 100000ULL);
    const ExperimentMetrics eps_015 = run_experiment(NUM_RUNS, NUM_ACTIONS, MAX_TIMESTEP, 0.15, 100000ULL);

    // Table Header
    std::cout << std::left
              << std::setw(6)  << "Step"
              << " | " << std::setw(15) << "Theoretical Opt"
              << " | " << std::setw(22) << "eps = 0.00 (Tot / Opt%)"
              << " | " << std::setw(22) << "eps = 0.01 (Tot / Opt%)"
              << " | " << std::setw(22) << "eps = 0.10 (Tot / Opt%)"
              << " | " << std::setw(22) << "eps = 0.15 (Tot / Opt%)"
              << "\n"
              << std::string(126, '-') << "\n";

    std::cout << std::fixed << std::setprecision(2);

    for (size_t t = 0; t <= MAX_TIMESTEP; ++t) {
        std::cout << std::left << std::setw(6) << t
                  << " | " << std::right << std::setw(15) << eps_000.avg_theoretical_optimal[t]
                  << " | " << std::setw(9) << eps_000.avg_cumulative_reward[t]
                  << " / " << std::setw(6) << eps_000.optimal_percentage[t] << "%"
                  << "   | " << std::setw(9) << eps_001.avg_cumulative_reward[t]
                  << " / " << std::setw(6) << eps_001.optimal_percentage[t] << "%"
                  << "   | " << std::setw(9) << eps_010.avg_cumulative_reward[t]
                  << " / " << std::setw(6) << eps_010.optimal_percentage[t] << "%"
                  << "   | " << std::setw(9) << eps_015.avg_cumulative_reward[t]
                  << " / " << std::setw(6) << eps_015.optimal_percentage[t] << "%"
                  << "\n";
    }

    // Peak Optimal Percentage Summary
    std::cout << std::string(126, '=') << "\n"
              << "Peak Optimal Performance:\n"
              << "  eps = 0.00 : " << eps_000.max_optimal_pct << "% at timestep " << eps_000.best_timestep << "\n"
              << "  eps = 0.01 : " << eps_001.max_optimal_pct << "% at timestep " << eps_001.best_timestep << "\n"
              << "  eps = 0.10 : " << eps_010.max_optimal_pct << "% at timestep " << eps_010.best_timestep << "\n"
              << "  eps = 0.15 : " << eps_015.max_optimal_pct << "% at timestep " << eps_015.best_timestep << "\n";

    return 0;
}