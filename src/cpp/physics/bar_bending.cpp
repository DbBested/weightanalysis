#include "bar_bending.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace weightlifting {

BarBendingModel::BarBendingModel()
    : length_(2.2),         // Standard Olympic barbell: 2.2m
      diameter_(0.028),     // 28mm diameter
      youngs_modulus_(200e9), // Steel: 200 GPa
      stiffness_(0.0),
      moment_of_inertia_(0.0) {

    // Compute moment of inertia for circular cross-section: I = π*r^4/4
    double radius = diameter_ / 2.0;
    moment_of_inertia_ = M_PI * std::pow(radius, 4) / 4.0;

    // Compute stiffness EI
    stiffness_ = youngs_modulus_ * moment_of_inertia_;
}

void BarBendingModel::setBarParameters(double length, double diameter, double youngs_modulus) {
    length_ = length;
    diameter_ = diameter;
    youngs_modulus_ = youngs_modulus;

    double radius = diameter_ / 2.0;
    moment_of_inertia_ = M_PI * std::pow(radius, 4) / 4.0;
    stiffness_ = youngs_modulus_ * moment_of_inertia_;
}

std::vector<double> BarBendingModel::computeDeflection(
    const std::vector<double>& force_positions,
    const std::vector<double>& force_magnitudes,
    int num_points) {

    std::vector<double> deflection(num_points, 0.0);

    if (force_positions.empty() || force_magnitudes.empty()) {
        return deflection;
    }

    // Compute deflection at each point along the bar
    for (int i = 0; i < num_points; ++i) {
        double x = static_cast<double>(i) / (num_points - 1); // Normalized position [0, 1]

        // Superposition of deflections from all forces
        for (size_t j = 0; j < force_positions.size() && j < force_magnitudes.size(); ++j) {
            deflection[i] += influenceFunction(x, force_positions[j], force_magnitudes[j]);
        }
    }

    return deflection;
}

std::vector<double> BarBendingModel::computeBendingMoment(
    const std::vector<double>& force_positions,
    const std::vector<double>& force_magnitudes,
    int num_points) {

    std::vector<double> moment(num_points, 0.0);

    if (force_positions.empty() || force_magnitudes.empty()) {
        return moment;
    }

    // Compute bending moment at each point
    for (int i = 0; i < num_points; ++i) {
        double x = static_cast<double>(i) / (num_points - 1);

        for (size_t j = 0; j < force_positions.size() && j < force_magnitudes.size(); ++j) {
            moment[i] += momentAtPosition(x, force_positions[j], force_magnitudes[j]);
        }
    }

    return moment;
}

std::pair<double, double> BarBendingModel::findMaxDeflection(
    const std::vector<double>& deflection) {

    if (deflection.empty()) {
        return {0.0, 0.0};
    }

    auto max_it = std::max_element(deflection.begin(), deflection.end(),
                                   [](double a, double b) { return std::abs(a) < std::abs(b); });

    double max_deflection = *max_it;
    double position = static_cast<double>(std::distance(deflection.begin(), max_it)) / deflection.size();

    return {max_deflection, position};
}

double BarBendingModel::computeStrainEnergy(const std::vector<double>& deflection) {
    if (deflection.size() < 2) {
        return 0.0;
    }

    // U = (1/2) * integral(M^2 / EI dx)
    // Approximate using trapezoidal rule
    double energy = 0.0;
    double dx = length_ / (deflection.size() - 1);

    for (size_t i = 1; i < deflection.size(); ++i) {
        // Approximate M from deflection: M = EI * d^2w/dx^2
        double d2w_dx2 = 0.0;
        if (i > 0 && i < deflection.size() - 1) {
            d2w_dx2 = (deflection[i+1] - 2*deflection[i] + deflection[i-1]) / (dx * dx);
        }

        double moment = stiffness_ * d2w_dx2;
        energy += 0.5 * moment * moment / stiffness_ * dx;
    }

    return energy;
}

std::vector<std::pair<int, int>> BarBendingModel::getCriticalRegions(
    const std::vector<double>& bending_moment,
    double threshold_fraction) {

    std::vector<std::pair<int, int>> regions;

    if (bending_moment.empty()) {
        return regions;
    }

    // Find max moment
    double max_moment = *std::max_element(bending_moment.begin(), bending_moment.end(),
                                         [](double a, double b) { return std::abs(a) < std::abs(b); });

    double threshold = threshold_fraction * std::abs(max_moment);

    // Find contiguous regions above threshold
    int start = -1;
    for (size_t i = 0; i < bending_moment.size(); ++i) {
        if (std::abs(bending_moment[i]) > threshold) {
            if (start == -1) {
                start = i;
            }
        } else {
            if (start != -1) {
                regions.push_back({start, static_cast<int>(i - 1)});
                start = -1;
            }
        }
    }

    // Close final region if needed
    if (start != -1) {
        regions.push_back({start, static_cast<int>(bending_moment.size() - 1)});
    }

    return regions;
}

double BarBendingModel::influenceFunction(double x, double force_pos, double force_mag) {
    // Simply supported beam deflection formula
    // w(x) = (F*a*b / 6*L*EI) * [x*(L^2 - x^2 - b^2)] for x < a
    // w(x) = (F*a*b / 6*L*EI) * [(L-x)*(x^2 - a^2 + 2*L*x - L^2)] for x >= a

    double L = 1.0; // Normalized length
    double a = force_pos;
    double b = L - a;
    double F = force_mag;

    double w = 0.0;

    if (x <= a) {
        w = (F * b / (6 * L * stiffness_)) * x * (L*L - x*x - b*b);
    } else {
        w = (F * a / (6 * L * stiffness_)) * (L - x) * (L*L - (L-x)*(L-x) - a*a);
    }

    return w * length_; // Scale to actual length
}

double BarBendingModel::momentAtPosition(double x, double force_pos, double force_mag) {
    // Bending moment for simply supported beam with point load
    // M(x) = F*a*(L-x)/L for x < a
    // M(x) = F*(L-a)*x/L for x >= a

    double L = 1.0; // Normalized
    double a = force_pos;
    double F = force_mag;

    double M = 0.0;

    if (x <= a) {
        M = F * a * (L - x) / L;
    } else {
        M = F * (L - a) * x / L;
    }

    return M * length_; // Scale to actual length
}

} // namespace weightlifting
