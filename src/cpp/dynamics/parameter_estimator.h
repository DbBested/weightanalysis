#pragma once

#include "multibody_model.h"
#include "inverse_dynamics.h"
#include <Eigen/Dense>
#include <vector>

namespace weightlifting {

// Parameter estimation using least squares or optimization
class ParameterEstimator {
public:
    ParameterEstimator();

    // Estimate bar mass from trajectory data
    // Uses physics constraints: F = ma
    double estimateBarMass(const std::vector<BarState>& bar_trajectory,
                           const std::vector<Eigen::Vector3d>& applied_forces);

    // Estimate lifter body segment masses
    std::vector<double> estimateSegmentMasses(
        MultibodyModel& model,
        const std::vector<JointState>& joint_trajectory,
        const std::vector<BarState>& bar_trajectory);

    // Physics-informed state estimation
    // Refines noisy pose estimates using physics constraints
    std::vector<JointState> refineStateEstimates(
        const std::vector<JointState>& noisy_states,
        const MultibodyModel& model,
        double process_noise = 0.01,
        double measurement_noise = 0.1);

    // Kalman filter for smooth state estimation
    JointState kalmanFilter(const JointState& measurement,
                           const JointState& prediction,
                           const Eigen::MatrixXd& process_cov,
                           const Eigen::MatrixXd& measurement_cov);

private:
    // Least squares solver for linear parameter estimation
    Eigen::VectorXd leastSquares(const Eigen::MatrixXd& A,
                                 const Eigen::VectorXd& b);

    // Weighted least squares with physics constraints
    Eigen::VectorXd constrainedLeastSquares(
        const Eigen::MatrixXd& A,
        const Eigen::VectorXd& b,
        const Eigen::MatrixXd& C,  // Constraint matrix
        const Eigen::VectorXd& d); // Constraint bounds
};

} // namespace weightlifting
