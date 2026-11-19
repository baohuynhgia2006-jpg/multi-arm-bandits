#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

class RandomGenerator {
public:
    // --- C++ UniformRandomBitGenerator (URBG) Interface ---
    // Enables direct compatibility with all <random> distributions and algorithms
    using result_type = uint64_t;
    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept { return std::numeric_limits<uint64_t>::max(); }
    result_type operator()() noexcept { return next_u64(); }

    // --- Construction & Seeding ---
    RandomGenerator();
    explicit RandomGenerator(uint64_t seed);
    void seed(uint64_t seed);
    void jump(); // Advances state by 2^128 calls for parallel streams

    // --- Core Uniform Generators ---
    uint64_t next_u64() noexcept;
    uint32_t next_u32() noexcept;
    double   next_double() noexcept;               // [0.0, 1.0)
    float    next_float() noexcept;                // [0.0f, 1.0f)
    double   uniform_real(double min, double max); // [min, max)
    int64_t  uniform_int(int64_t min, int64_t max); // [min, max] (unbiased)

    // --- Discrete Distributions ---
    bool    bernoulli(double p = 0.5);
    int64_t binomial(int64_t n, double p);
    int64_t geometric(double p);
    int64_t negative_binomial(int64_t r, double p);
    int64_t poisson(double lambda);
    size_t  categorical(const std::vector<double>& weights);

    // --- Continuous Distributions ---
    double normal(double mean = 0.0, double stddev = 1.0);
    double log_normal(double mean = 0.0, double stddev = 1.0);
    double exponential(double lambda = 1.0);
    double gamma(double alpha, double beta = 1.0);
    double beta(double alpha, double beta);
    double chi_squared(double k);
    double student_t(double nu);
    double cauchy(double location = 0.0, double scale = 1.0);
    double weibull(double shape, double scale = 1.0);

private:
    std::array<uint64_t, 4> state_{};
    bool has_spare_normal_ = false;
    double spare_normal_ = 0.0;
};

#endif // RANDOM_GENERATOR_H