#include "parameter_estimator.h"
#include <cmath>
#include <limits>

namespace weightlifting {

ParameterEstimator::ParameterEstimator() {
}

double ParameterEstimator::estimateBarMass(
    const std::vector<BarState>& bar_trajectory,
    const std::vector<Eigen::Vector3d>& applied_forces) {

    if (bar_trajectory.size() < 2 || applied_forces.empty()) {
        return 20.0; // Default barbell mass
    }

    // Use F = ma to estimate mass
    // m = F / a
    std::vector<double> mass_estimates;

    for (size_t i = 0; i < bar_trajectory.size() && i < applied_forces.size(); ++i) {
        const auto& state = bar_trajectory[i];
        const auto& force = applied_forces[i];

        // Only use points with significant acceleration
        double accel_mag = state.acceleration.norm();
        if (accel_mag > 0.5) { // m/s^2 threshold
            // F - mg = ma, so m = F / (a + g)
            double force_mag = force.norm();
            double total_accel = accel_mag + 9.81; // Account for gravity

            if (total_accel > 0.1) {
                double mass_est = force_mag / total_accel;
                if (mass_est > 10.0 && mass_est < 300.0) { // Reasonable range
                    mass_estimates.push_back(mass_est);
                }
            }
        }
    }

    // Return median estimate (robust to outliers)
    if (mass_estimates.empty()) {
        return 20.0;
    }

    std::sort(mass_estimates.begin(), mass_estimates.end());
    return mass_estimates[mass_estimates.size() / 2];
}

std::vector<double> ParameterEstimator::estimateSegmentMasses(
    MultibodyModel& model,
    const std::vector<JointState>& joint_trajectory,
    const std::vector<BarState>& bar_trajectory) {

    const auto& segments = model.getSegments();
    std::vector<double> estimated_masses(segments.size());

    // Use anthropometric regression models as priors
    // Then refine using inverse dynamics residuals

    for (size_t i = 0; i < segments.size(); ++i) {
        estimated_masses[i] = segments[i].mass; // Start with defaults
    }

    // Iterative refinement using inverse dynamics
    // This is simplified; full implementation would use optimization
    InverseDynamicsSolver solver(model);

    for (int iter = 0; iter < 3; ++iter) {
        // Compute residuals
        auto result = solver.solve(joint_trajectory, bar_trajectory);

        // Adjust masses based on force residuals
        // (Simplified heuristic)
        for (size_t i = 0; i < estimated_masses.size() && i < result.bar_forces.size(); ++i) {
            double force_error = result.bar_forces[i].norm() - 500.0; // Expected force
            estimated_masses[i] *= (1.0 + force_error * 0.0001); // Small adjustment
            estimated_masses[i] = std::max(1.0, std::min(50.0, estimated_masses[i]));
        }

        // Update model
        for (size_t i = 0; i < estimated_masses.size(); ++i) {
            model.updateSegmentMass(i, estimated_masses[i]);
        }
    }

    return estimated_masses;
}

std::vector<JointState> ParameterEstimator::refineStateEstimates(
    const std::vector<JointState>& noisy_states,
    const MultibodyModel& model,
    double process_noise,
    double measurement_noise) {

    if (noisy_states.empty()) {
        return noisy_states;
    }

    std::vector<JointState> refined_states;
    refined_states.reserve(noisy_states.size());

    int dof = noisy_states[0].positions.size();

    // Initialize Kalman filter covariances
    Eigen::MatrixXd process_cov = process_noise * Eigen::MatrixXd::Identity(dof * 2, dof * 2);
    Eigen::MatrixXd measurement_cov = measurement_noise * Eigen::MatrixXd::Identity(dof, dof);

    JointState predicted = noisy_states[0];
    refined_states.push_back(predicted);

    // Apply Kalman filter
    for (size_t i = 1; i < noisy_states.size(); ++i) {
        // Predict step (constant velocity model)
        double dt = noisy_states[i].timestamp - noisy_states[i-1].timestamp;
        if (dt <= 0) dt = 0.033; // ~30 fps default

        JointState prediction(dof);
        prediction.positions = predicted.positions + predicted.velocities * dt;
        prediction.velocities = predicted.velocities;
        prediction.timestamp = noisy_states[i].timestamp;

        // Update step
        predicted = kalmanFilter(noisy_states[i], prediction, process_cov, measurement_cov);
        refined_states.push_back(predicted);
    }

    // Smooth velocities and compute accelerations
    for (size_t i = 1; i < refined_states.size() - 1; ++i) {
        double dt = refined_states[i+1].timestamp - refined_states[i-1].timestamp;
        if (dt > 0) {
            refined_states[i].velocities = (refined_states[i+1].positions - refined_states[i-1].positions) / dt;

            // Central difference for acceleration
            if (i > 0) {
                double dt2 = refined_states[i].timestamp - refined_states[i-1].timestamp;
                if (dt2 > 0) {
                    refined_states[i].accelerations = (refined_states[i].velocities - refined_states[i-1].velocities) / dt2;
                }
            }
        }
    }

    return refined_states;
}

JointState ParameterEstimator::kalmanFilter(
    const JointState& measurement,
    const JointState& prediction,
    const Eigen::MatrixXd& process_cov,
    const Eigen::MatrixXd& measurement_cov) {

    int dof = measurement.positions.size();

    // Simplified Kalman filter for positions only
    Eigen::MatrixXd K = process_cov.block(0, 0, dof, dof) *
                       (process_cov.block(0, 0, dof, dof) + measurement_cov).inverse();

    JointState filtered = prediction;
    filtered.positions = prediction.positions + K * (measurement.positions - prediction.positions);

    // Simple velocity update
    double dt = measurement.timestamp - prediction.timestamp;
    if (dt > 0) {
        filtered.velocities = (filtered.positions - prediction.positions) / dt;
    }

    return filtered;
}

Eigen::VectorXd ParameterEstimator::leastSquares(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b) {

    // Solve Ax = b using normal equations: x = (A^T A)^-1 A^T b
    return (A.transpose() * A).ldlt().solve(A.transpose() * b);
}

Eigen::VectorXd ParameterEstimator::constrainedLeastSquares(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& d) {

    // Solve min ||Ax - b||^2 subject to Cx >= d
    // Using Lagrangian method (simplified)

    Eigen::VectorXd x_unconstrained = leastSquares(A, b);

    // Check constraints
    Eigen::VectorXd constraint_residual = C * x_unconstrained - d;
    bool all_satisfied = (constraint_residual.array() >= 0).all();

    if (all_satisfied) {
        return x_unconstrained;
    }

    // Project onto feasible region (simplified)
    Eigen::VectorXd x_constrained = x_unconstrained;
    for (int i = 0; i < constraint_residual.size(); ++i) {
        if (constraint_residual(i) < 0) {
            // Simple projection
            x_constrained += C.row(i).transpose() * (-constraint_residual(i));
        }
    }

    return x_constrained;
}

} // namespace weightlifting
