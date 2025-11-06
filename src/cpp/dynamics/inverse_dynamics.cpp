#include "inverse_dynamics.h"
#include <cmath>
#include <algorithm>

namespace weightlifting {

InverseDynamicsSolver::InverseDynamicsSolver(const MultibodyModel& model)
    : model_(model), external_force_(Eigen::Vector3d(0, -9.81, 0)) {
}

InverseDynamicsResult InverseDynamicsSolver::solve(
    const std::vector<JointState>& trajectory,
    const std::vector<BarState>& bar_trajectory) {

    InverseDynamicsResult result;
    result.timestamps.reserve(trajectory.size());
    result.joint_torques.reserve(trajectory.size());
    result.bar_forces.reserve(trajectory.size());
    result.reaction_forces.reserve(trajectory.size());
    result.power.reserve(trajectory.size());

    result.peak_force = 0.0;
    result.peak_time = 0.0;

    // Process each time step
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const JointState& state = trajectory[i];
        const BarState& bar_state = (i < bar_trajectory.size()) ? bar_trajectory[i] : BarState();

        // Compute inverse dynamics
        Eigen::VectorXd torques = computeInverseDynamicsRNE(state, bar_state);
        result.joint_torques.push_back(torques);

        // Compute forces
        Eigen::Vector3d bar_force = computeBarForce(state, bar_state, torques);
        Eigen::Vector3d reaction_force = computeGroundReactionForce(state, torques);

        result.bar_forces.push_back(bar_force);
        result.reaction_forces.push_back(reaction_force);
        result.timestamps.push_back(state.timestamp);

        // Compute power
        double power = computePower(state, torques);
        result.power.push_back(power);

        // Track peak force
        double force_mag = bar_force.norm();
        if (force_mag > result.peak_force) {
            result.peak_force = force_mag;
            result.peak_time = state.timestamp;
        }
    }

    return result;
}

Eigen::VectorXd InverseDynamicsSolver::computeInverseDynamicsRNE(
    const JointState& state,
    const BarState& bar_state) {

    int dof = model_.getDegreesOfFreedom();
    Eigen::VectorXd torques = Eigen::VectorXd::Zero(dof);

    // Standard inverse dynamics equation: tau = M(q)*qdd + C(q,qd) + G(q)
    Eigen::MatrixXd M = model_.computeMassMatrix(state);
    Eigen::VectorXd C = model_.computeCoriolisForces(state);
    Eigen::VectorXd G = model_.computeGravityForces(state);

    torques = M * state.accelerations + C + G;

    // Add contribution from bar dynamics
    // F_bar = m_bar * a_bar + m_bar * g
    double bar_mass = 20.0; // Default, should be from model
    Eigen::Vector3d bar_force = bar_mass * (bar_state.acceleration + Eigen::Vector3d(0, 9.81, 0));

    // Map bar force to joint torques through Jacobian transpose
    if (dof > 6) {
        Eigen::MatrixXd J = model_.computeJacobian(state, 6); // End-effector (hands)
        Eigen::VectorXd joint_contribution = J.transpose() * bar_force;

        // Add to torques
        for (int i = 0; i < std::min((int)joint_contribution.size(), dof); ++i) {
            torques(i) += joint_contribution(i);
        }
    }

    return torques;
}

Eigen::Vector3d InverseDynamicsSolver::computeGroundReactionForce(
    const JointState& state,
    const Eigen::VectorXd& joint_torques) {

    // Ground reaction force is the sum of all body weights plus acceleration forces
    Eigen::Vector3d grf = Eigen::Vector3d::Zero();

    const auto& segments = model_.getSegments();
    double total_mass = 0.0;

    for (const auto& seg : segments) {
        total_mass += seg.mass;
    }

    // Static component (weight)
    grf(1) = total_mass * 9.81;

    // Dynamic component from joint accelerations
    // Simplified: assume vertical acceleration proportional to average joint acceleration
    double avg_accel = 0.0;
    if (state.accelerations.size() > 0) {
        avg_accel = state.accelerations.sum() / state.accelerations.size();
    }
    grf(1) += total_mass * avg_accel * 0.5; // Heuristic scaling

    return grf;
}

Eigen::Vector3d InverseDynamicsSolver::computeBarForce(
    const JointState& state,
    const BarState& bar_state,
    const Eigen::VectorXd& joint_torques) {

    // Force on bar from lifter's hands
    // F = m*a + m*g (Newton's second law)
    double bar_mass = 20.0; // kg, should come from model or estimation

    Eigen::Vector3d gravity_force(0, bar_mass * 9.81, 0);
    Eigen::Vector3d inertial_force = bar_mass * bar_state.acceleration;

    Eigen::Vector3d bar_force = inertial_force + gravity_force;

    return bar_force;
}

double InverseDynamicsSolver::computePower(
    const JointState& state,
    const Eigen::VectorXd& joint_torques) {

    // Power = tau^T * q_dot
    double power = 0.0;

    int n = std::min((int)joint_torques.size(), (int)state.velocities.size());
    for (int i = 0; i < n; ++i) {
        power += joint_torques(i) * state.velocities(i);
    }

    return power;
}

void InverseDynamicsSolver::setExternalForce(const Eigen::Vector3d& force) {
    external_force_ = force;
}

void InverseDynamicsSolver::forwardPass(
    const JointState& state,
    std::vector<Eigen::Vector3d>& link_velocities,
    std::vector<Eigen::Vector3d>& link_accelerations) {

    // Propagate velocities and accelerations from base to tip
    const auto& segments = model_.getSegments();
    link_velocities.resize(segments.size());
    link_accelerations.resize(segments.size());

    for (size_t i = 0; i < segments.size(); ++i) {
        if (i < (size_t)state.velocities.size()) {
            link_velocities[i] = Eigen::Vector3d(0, state.velocities(i), 0);
            link_accelerations[i] = Eigen::Vector3d(0, state.accelerations(i), 0);
        }
    }
}

void InverseDynamicsSolver::backwardPass(
    const JointState& state,
    const std::vector<Eigen::Vector3d>& link_accelerations,
    Eigen::VectorXd& torques) {

    // Propagate forces from tip to base
    const auto& segments = model_.getSegments();

    for (int i = segments.size() - 1; i >= 0; --i) {
        if (i < torques.size()) {
            // Simplified: torque = I * alpha + r x F
            double I = segments[i].mass * segments[i].length * segments[i].length / 12.0;
            torques(i) = I * link_accelerations[i](1);
        }
    }
}

} // namespace weightlifting
