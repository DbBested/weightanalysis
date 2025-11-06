#pragma once

#include "multibody_model.h"
#include <Eigen/Dense>
#include <vector>

namespace weightlifting {

// Results from inverse dynamics computation
struct InverseDynamicsResult {
    std::vector<Eigen::VectorXd> joint_torques;  // Computed torques at each time step
    std::vector<Eigen::Vector3d> bar_forces;     // Forces on the bar
    std::vector<Eigen::Vector3d> reaction_forces; // Ground reaction forces
    std::vector<double> timestamps;
    double peak_force;                           // Maximum force magnitude
    double peak_time;                            // Time of peak force
    std::vector<double> power;                   // Power output at each time step
};

// Inverse dynamics solver
class InverseDynamicsSolver {
public:
    explicit InverseDynamicsSolver(const MultibodyModel& model);

    // Main inverse dynamics computation
    // Given joint states over time, compute required torques and forces
    InverseDynamicsResult solve(const std::vector<JointState>& trajectory,
                                const std::vector<BarState>& bar_trajectory);

    // Recursive Newton-Euler algorithm
    Eigen::VectorXd computeInverseDynamicsRNE(const JointState& state,
                                              const BarState& bar_state);

    // Compute ground reaction forces
    Eigen::Vector3d computeGroundReactionForce(const JointState& state,
                                               const Eigen::VectorXd& joint_torques);

    // Compute bar forces from lifter's hands
    Eigen::Vector3d computeBarForce(const JointState& state,
                                    const BarState& bar_state,
                                    const Eigen::VectorXd& joint_torques);

    // Compute mechanical power
    double computePower(const JointState& state,
                       const Eigen::VectorXd& joint_torques);

    // Set external force (e.g., bar weight)
    void setExternalForce(const Eigen::Vector3d& force);

private:
    const MultibodyModel& model_;
    Eigen::Vector3d external_force_;

    // Forward pass: compute velocities and accelerations
    void forwardPass(const JointState& state,
                    std::vector<Eigen::Vector3d>& link_velocities,
                    std::vector<Eigen::Vector3d>& link_accelerations);

    // Backward pass: compute forces and torques
    void backwardPass(const JointState& state,
                     const std::vector<Eigen::Vector3d>& link_accelerations,
                     Eigen::VectorXd& torques);
};

} // namespace weightlifting
