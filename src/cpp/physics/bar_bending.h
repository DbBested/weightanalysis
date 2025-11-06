#pragma once

#include <Eigen/Dense>
#include <vector>

namespace weightlifting {

// Bar bending analysis using beam theory
class BarBendingModel {
public:
    BarBendingModel();

    // Initialize bar parameters
    void setBarParameters(double length, double diameter, double youngs_modulus);

    // Compute bar deflection given applied forces and positions
    // Returns deflection at each point along the bar
    std::vector<double> computeDeflection(
        const std::vector<double>& force_positions,  // Normalized positions [0, 1]
        const std::vector<double>& force_magnitudes, // Forces in N
        int num_points = 50);                        // Resolution

    // Compute bending moment distribution
    std::vector<double> computeBendingMoment(
        const std::vector<double>& force_positions,
        const std::vector<double>& force_magnitudes,
        int num_points = 50);

    // Find maximum deflection and its location
    std::pair<double, double> findMaxDeflection(const std::vector<double>& deflection);

    // Compute strain energy in the bar
    double computeStrainEnergy(const std::vector<double>& deflection);

    // Get critical bending regions (where |M| > threshold)
    std::vector<std::pair<int, int>> getCriticalRegions(
        const std::vector<double>& bending_moment,
        double threshold_fraction = 0.7);

    // Getters
    double getLength() const { return length_; }
    double getStiffness() const { return stiffness_; }

private:
    double length_;           // Bar length (m)
    double diameter_;         // Bar diameter (m)
    double youngs_modulus_;   // Young's modulus (Pa)
    double stiffness_;        // EI (N*m^2)
    double moment_of_inertia_; // Second moment of area (m^4)

    // Compute influence function for beam deflection
    double influenceFunction(double x, double force_pos, double force_mag);

    // Compute moment at position x due to force at force_pos
    double momentAtPosition(double x, double force_pos, double force_mag);
};

} // namespace weightlifting
