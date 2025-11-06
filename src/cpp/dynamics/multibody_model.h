#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace weightlifting {

// Represents a single body segment (limb, torso, bar, etc.)
struct BodySegment {
    std::string name;
    double mass;                    // kg
    double length;                  // m
    Eigen::Vector3d com_offset;     // Center of mass offset from proximal joint
    Eigen::Matrix3d inertia;        // Moment of inertia tensor

    BodySegment() : mass(0.0), length(0.0), com_offset(Eigen::Vector3d::Zero()),
                    inertia(Eigen::Matrix3d::Identity()) {}
};

// Joint state at a single time point
struct JointState {
    Eigen::VectorXd positions;      // Joint angles/positions
    Eigen::VectorXd velocities;     // Joint velocities
    Eigen::VectorXd accelerations;  // Joint accelerations
    double timestamp;               // seconds

    JointState(int dof = 0) : positions(Eigen::VectorXd::Zero(dof)),
                              velocities(Eigen::VectorXd::Zero(dof)),
                              accelerations(Eigen::VectorXd::Zero(dof)),
                              timestamp(0.0) {}
};

// Bar state with position and bending
struct BarState {
    Eigen::Vector3d position;       // Center position
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    Eigen::Quaterniond orientation;
    double bending_moment;          // N*m
    std::vector<double> deflection; // Deflection along bar length

    BarState() : position(Eigen::Vector3d::Zero()),
                 velocity(Eigen::Vector3d::Zero()),
                 acceleration(Eigen::Vector3d::Zero()),
                 orientation(Eigen::Quaterniond::Identity()),
                 bending_moment(0.0) {}
};

// Multibody model for the lifter + bar system
class MultibodyModel {
public:
    MultibodyModel();

    // Initialize model for different lift types
    void initializeSquatModel();
    void initializeBenchModel();
    void initializeDeadliftModel();

    // Add body segments
    void addSegment(const BodySegment& segment);

    // Compute mass matrix M(q)
    Eigen::MatrixXd computeMassMatrix(const JointState& state) const;

    // Compute Coriolis and centrifugal forces C(q, q_dot)
    Eigen::VectorXd computeCoriolisForces(const JointState& state) const;

    // Compute gravitational forces G(q)
    Eigen::VectorXd computeGravityForces(const JointState& state) const;

    // Forward kinematics: compute end-effector position
    Eigen::Vector3d computeEndEffectorPosition(const JointState& state, int chain_id) const;

    // Compute Jacobian matrix
    Eigen::MatrixXd computeJacobian(const JointState& state, int chain_id) const;

    // Getters
    int getDegreesOfFreedom() const { return dof_; }
    const std::vector<BodySegment>& getSegments() const { return segments_; }

    // Setters for parameter estimation
    void updateSegmentMass(int segment_id, double mass);
    void updateBarMass(double mass);

private:
    std::vector<BodySegment> segments_;
    int dof_;  // Total degrees of freedom
    double bar_mass_;
    double gravity_;

    // Helper functions
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) const;
};

} // namespace weightlifting
