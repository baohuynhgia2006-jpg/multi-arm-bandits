#include "..\header\RandomGenerator.h"
#include <cmath>
#include <numeric>
#include <random>

namespace {
    inline uint64_t rotl(const uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t splitmix64(uint64_t& x) noexcept {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Standard normal N(0, 1) using Marsaglia polar method
    double standard_normal(RandomGenerator& rng, bool& has_spare, double& spare) noexcept {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        double u, v, s;
        do {
            u = rng.next_double() * 2.0 - 1.0;
            v = rng.next_double() * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        const double mul = std::sqrt(-2.0 * std::log(s) / s);
        spare = v * mul;
        has_spare = true;
        return u * mul;
    }

    // Standard gamma Gamma(alpha, 1.0) using Marsaglia & Tsang's method
    double standard_gamma(RandomGenerator& rng, double alpha, bool& has_spare, double& spare) noexcept {
        if (alpha < 1.0) {
            // Boost alpha < 1 using: Gamma(alpha) = Gamma(alpha + 1) * U^(1/alpha)
            const double u = 1.0 - rng.next_double(); // (0, 1]
            return standard_gamma(rng, alpha + 1.0, has_spare, spare) * std::pow(u, 1.0 / alpha);
        }

        const double d = alpha - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        while (true) {
            double x, v;
            do {
                x = standard_normal(rng, has_spare, spare);
                v = 1.0 + c * x;
            } while (v <= 0.0);

            v = v * v * v;
            const double u = 1.0 - rng.next_double(); // (0, 1]
            if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
                return d * v;
            }
            if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
                return d * v;
            }
        }
    }
} // namespace

// --- Construction & Seeding ---

RandomGenerator::RandomGenerator() {
    std::random_device rd;
    uint64_t initial_seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    seed(initial_seed);
}

RandomGenerator::RandomGenerator(uint64_t initial_seed) {
    seed(initial_seed);
}

void RandomGenerator::seed(uint64_t initial_seed) {
    has_spare_normal_ = false;
    for (uint64_t& s : state_) {
        s = splitmix64(initial_seed);
    }
}

void RandomGenerator::jump() {
    static constexpr uint64_t JUMP[] = {
        0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
        0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
    };

    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (uint64_t j : JUMP) {
        for (int b = 0; b < 64; b++) {
            if (j & (1ULL << b)) {
                s0 ^= state_[0];
                s1 ^= state_[1];
                s2 ^= state_[2];
                s3 ^= state_[3];
            }
            next_u64();
        }
    }
    state_ = {s0, s1, s2, s3};
}

// --- Core Uniform Generators ---

uint64_t RandomGenerator::next_u64() noexcept {
    const uint64_t result = rotl(state_[1] * 5, 7) * 9;
    const uint64_t t = state_[1] << 17;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];

    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);

    return result;
}

uint32_t RandomGenerator::next_u32() noexcept {
    return static_cast<uint32_t>(next_u64() >> 32);
}

double RandomGenerator::next_double() noexcept {
    return (next_u64() >> 11) * 0x1.0p-53;
}

float RandomGenerator::next_float() noexcept {
    return (next_u64() >> 40) * 0x1.0p-24f;
}

double RandomGenerator::uniform_real(double min, double max) {
    if (min >= max) {
        throw std::invalid_argument("RandomGenerator::uniform_real: min must be less than max");
    }
    return min + next_double() * (max - min);
}

int64_t RandomGenerator::uniform_int(int64_t min, int64_t max) {
    if (min > max) {
        throw std::invalid_argument("RandomGenerator::uniform_int: min cannot be greater than max");
    }
    const uint64_t range = static_cast<uint64_t>(max) - static_cast<uint64_t>(min) + 1ULL;
    if (range == 0) {
        return static_cast<int64_t>(next_u64());
    }

    // Standard 64-bit rejection sampling (no 128-bit types)
    const uint64_t threshold = (-range) % range;
    uint64_t r;
    do {
        r = next_u64();
    } while (r < threshold);

    return min + static_cast<int64_t>(r % range);
}

// --- Discrete Distributions ---

bool RandomGenerator::bernoulli(double p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("RandomGenerator::bernoulli: p must be in [0.0, 1.0]");
    }
    if (p == 0.0) return false;
    if (p == 1.0) return true;
    return next_double() < p;
}

int64_t RandomGenerator::binomial(int64_t n, double p) {
    if (n < 0 || p < 0.0 || p > 1.0) {
        throw std::invalid_argument("RandomGenerator::binomial: requires n >= 0 and p in [0.0, 1.0]");
    }
    if (n == 0 || p == 0.0) return 0;
    if (p == 1.0) return n;

    // For large n, use normal approximation with continuity correction
    if (n > 1000 && n * p > 10.0 && n * (1.0 - p) > 10.0) {
        const double mean = n * p;
        const double stddev = std::sqrt(n * p * (1.0 - p));
        int64_t val = static_cast<int64_t>(std::round(normal(mean, stddev)));
        if (val < 0) return 0;
        if (val > n) return n;
        return val;
    }

    // Exact geometric waiting-time algorithm (Devroye)
    const double q = (p <= 0.5) ? p : (1.0 - p);
    const double log_q = std::log(1.0 - q);
    int64_t x = 0;
    int64_t sum = 0;
    while (true) {
        const double u = 1.0 - next_double();
        sum += static_cast<int64_t>(std::floor(std::log(u) / log_q)) + 1;
        if (sum > n) break;
        x++;
    }
    return (p <= 0.5) ? x : (n - x);
}

int64_t RandomGenerator::geometric(double p) {
    if (p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("RandomGenerator::geometric: p must be in (0.0, 1.0]");
    }
    if (p == 1.0) return 0; // Number of failures before first success
    const double u = 1.0 - next_double();
    return static_cast<int64_t>(std::floor(std::log(u) / std::log(1.0 - p)));
}

int64_t RandomGenerator::negative_binomial(int64_t r, double p) {
    if (r <= 0 || p <= 0.0 || p > 1.0) {
        throw std::invalid_argument("RandomGenerator::negative_binomial: requires r > 0 and p in (0.0, 1.0]");
    }
    if (p == 1.0) return 0;
    // Gamma-Poisson mixture representation
    const double lambda = gamma(static_cast<double>(r), (1.0 - p) / p);
    return poisson(lambda);
}

int64_t RandomGenerator::poisson(double lambda) {
    if (lambda < 0.0) {
        throw std::invalid_argument("RandomGenerator::poisson: lambda must be >= 0.0");
    }
    if (lambda == 0.0) return 0;

    // Knuth's algorithm for small lambda
    if (lambda < 30.0) {
        const double L = std::exp(-lambda);
        int64_t k = 0;
        double p = 1.0;
        do {
            k++;
            p *= next_double();
        } while (p > L);
        return k - 1;
    }

    // Atkinson's / Rejection method (PTRS) for large lambda
    const double c = 0.767 - 3.36 / lambda;
    const double beta = 3.14159265358979323846 / std::sqrt(3.0 * lambda);
    const double alpha = beta * lambda;
    const double k = std::log(c) - lambda - std::log(beta);

    while (true) {
        const double u = next_double();
        if (u == 0.0) continue;
        const double x = (alpha - std::log((1.0 - u) / u)) / beta;
        const int64_t n = static_cast<int64_t>(std::floor(x + 0.5));
        if (n < 0) continue;
        const double v = next_double();
        if (v == 0.0) continue;
        const double y = alpha - beta * x;
        const double lhs = y + std::log(v / ((1.0 + std::exp(y)) * (1.0 + std::exp(y))));
        const double rhs = k + n * std::log(lambda) - std::lgamma(n + 1.0);
        if (lhs <= rhs) return n;
    }
}

size_t RandomGenerator::categorical(const std::vector<double>& weights) {
    if (weights.empty()) {
        throw std::invalid_argument("RandomGenerator::categorical: weights cannot be empty");
    }
    double total = 0.0;
    for (double w : weights) {
        if (w < 0.0) throw std::invalid_argument("RandomGenerator::categorical: weights must be >= 0");
        total += w;
    }
    if (total <= 0.0) {
        throw std::invalid_argument("RandomGenerator::categorical: sum of weights must be > 0");
    }

    const double target = next_double() * total;
    double cumulative = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (target < cumulative) {
            return i;
        }
    }
    return weights.size() - 1;
}

// --- Continuous Distributions ---

double RandomGenerator::normal(double mean, double stddev) {
    if (stddev < 0.0) {
        throw std::invalid_argument("RandomGenerator::normal: stddev must be >= 0.0");
    }
    return mean + stddev * standard_normal(*this, has_spare_normal_, spare_normal_);
}

double RandomGenerator::log_normal(double mean, double stddev) {
    return std::exp(normal(mean, stddev));
}

double RandomGenerator::exponential(double lambda) {
    if (lambda <= 0.0) {
        throw std::invalid_argument("RandomGenerator::exponential: lambda must be > 0.0");
    }
    const double u = 1.0 - next_double(); // (0, 1] avoids log(0)
    return -std::log(u) / lambda;
}

double RandomGenerator::gamma(double alpha, double beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("RandomGenerator::gamma: alpha and beta must be > 0.0");
    }
    // Scale parameterization: beta is scale (mean = alpha * beta)
    return standard_gamma(*this, alpha, has_spare_normal_, spare_normal_) * beta;
}

double RandomGenerator::beta(double alpha, double beta) {
    if (alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("RandomGenerator::beta: alpha and beta must be > 0.0");
    }
    const double x = standard_gamma(*this, alpha, has_spare_normal_, spare_normal_);
    const double y = standard_gamma(*this, beta, has_spare_normal_, spare_normal_);
    return x / (x + y);
}

double RandomGenerator::chi_squared(double k) {
    if (k <= 0.0) {
        throw std::invalid_argument("RandomGenerator::chi_squared: degrees of freedom k must be > 0.0");
    }
    return gamma(k * 0.5, 2.0);
}

double RandomGenerator::student_t(double nu) {
    if (nu <= 0.0) {
        throw std::invalid_argument("RandomGenerator::student_t: degrees of freedom nu must be > 0.0");
    }
    const double z = standard_normal(*this, has_spare_normal_, spare_normal_);
    const double v = chi_squared(nu);
    return z / std::sqrt(v / nu);
}

double RandomGenerator::cauchy(double location, double scale) {
    if (scale <= 0.0) {
        throw std::invalid_argument("RandomGenerator::cauchy: scale must be > 0.0");
    }
    constexpr double pi = 3.14159265358979323846;
    double u;
    do {
        u = next_double();
    } while (u == 0.0);
    return location + scale * std::tan(pi * (u - 0.5));
}

double RandomGenerator::weibull(double shape, double scale) {
    if (shape <= 0.0 || scale <= 0.0) {
        throw std::invalid_argument("RandomGenerator::weibull: shape and scale must be > 0.0");
    }
    const double u = 1.0 - next_double(); // (0, 1]
    return scale * std::pow(-std::log(u), 1.0 / shape);
}