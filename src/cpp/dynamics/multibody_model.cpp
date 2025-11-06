#include "multibody_model.h"
#include <cmath>

namespace weightlifting {

MultibodyModel::MultibodyModel()
    : dof_(0), bar_mass_(20.0), gravity_(9.81) {
}

void MultibodyModel::initializeSquatModel() {
    segments_.clear();
    dof_ = 12; // 6 for lower body, 6 for upper body/arms

    // Anthropometric data (average male)
    // Thigh segments
    BodySegment thigh_r, thigh_l;
    thigh_r.name = "thigh_right";
    thigh_r.mass = 10.5;  // kg
    thigh_r.length = 0.43; // m
    thigh_r.com_offset = Eigen::Vector3d(0, -0.43*0.433, 0);
    thigh_l = thigh_r;
    thigh_l.name = "thigh_left";

    // Shank segments
    BodySegment shank_r, shank_l;
    shank_r.name = "shank_right";
    shank_r.mass = 3.5;
    shank_r.length = 0.43;
    shank_r.com_offset = Eigen::Vector3d(0, -0.43*0.433, 0);
    shank_l = shank_r;
    shank_l.name = "shank_left";

    // Torso
    BodySegment torso;
    torso.name = "torso";
    torso.mass = 35.0;
    torso.length = 0.60;
    torso.com_offset = Eigen::Vector3d(0, 0.30, 0);

    // Arms (combined for simplicity)
    BodySegment arms;
    arms.name = "arms";
    arms.mass = 8.0;
    arms.length = 0.60;
    arms.com_offset = Eigen::Vector3d(0, -0.30, 0);

    segments_.push_back(thigh_r);
    segments_.push_back(thigh_l);
    segments_.push_back(shank_r);
    segments_.push_back(shank_l);
    segments_.push_back(torso);
    segments_.push_back(arms);
}

void MultibodyModel::initializeBenchModel() {
    segments_.clear();
    dof_ = 10;

    // Upper body focused model
    BodySegment torso;
    torso.name = "torso";
    torso.mass = 40.0;
    torso.length = 0.60;

    BodySegment upper_arm_r, upper_arm_l;
    upper_arm_r.name = "upper_arm_right";
    upper_arm_r.mass = 2.5;
    upper_arm_r.length = 0.30;
    upper_arm_l = upper_arm_r;
    upper_arm_l.name = "upper_arm_left";

    BodySegment forearm_r, forearm_l;
    forearm_r.name = "forearm_right";
    forearm_r.mass = 1.5;
    forearm_r.length = 0.27;
    forearm_l = forearm_r;
    forearm_l.name = "forearm_left";

    segments_.push_back(torso);
    segments_.push_back(upper_arm_r);
    segments_.push_back(upper_arm_l);
    segments_.push_back(forearm_r);
    segments_.push_back(forearm_l);
}

void MultibodyModel::initializeDeadliftModel() {
    initializeSquatModel(); // Similar model, different initial pose
}

void MultibodyModel::addSegment(const BodySegment& segment) {
    segments_.push_back(segment);
}

Eigen::MatrixXd MultibodyModel::computeMassMatrix(const JointState& state) const {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dof_, dof_);

    // Simplified mass matrix computation
    // In a full implementation, this would use recursive Newton-Euler or composite rigid body algorithm
    for (size_t i = 0; i < segments_.size() && i < (size_t)dof_; ++i) {
        M(i, i) = segments_[i].mass * segments_[i].length * segments_[i].length / 3.0;
    }

    // Add bar mass contribution (acts at hands)
    if (dof_ > 6) {
        M(6, 6) += bar_mass_;
    }

    return M;
}

Eigen::VectorXd MultibodyModel::computeCoriolisForces(const JointState& state) const {
    Eigen::VectorXd C = Eigen::VectorXd::Zero(dof_);

    // Simplified Coriolis computation
    // C = h(q, q_dot) where h contains Coriolis and centrifugal terms
    for (int i = 0; i < dof_ && i < (int)segments_.size(); ++i) {
        double q = state.positions(i);
        double qd = state.velocities(i);
        // Simplified centrifugal term
        C(i) = -segments_[i].mass * segments_[i].length * qd * qd * std::sin(q) * std::cos(q);
    }

    return C;
}

Eigen::VectorXd MultibodyModel::computeGravityForces(const JointState& state) const {
    Eigen::VectorXd G = Eigen::VectorXd::Zero(dof_);

    // Gravity acts downward on each segment's center of mass
    for (int i = 0; i < dof_ && i < (int)segments_.size(); ++i) {
        double q = state.positions(i);
        G(i) = segments_[i].mass * gravity_ * segments_[i].length * std::cos(q) / 2.0;
    }

    return G;
}

Eigen::Vector3d MultibodyModel::computeEndEffectorPosition(const JointState& state, int chain_id) const {
    // Forward kinematics using DH parameters or direct computation
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();

    // Simplified FK for demonstration
    for (int i = 0; i <= chain_id && i < (int)segments_.size(); ++i) {
        double q = state.positions(i);
        pos(0) += segments_[i].length * std::cos(q);
        pos(1) += segments_[i].length * std::sin(q);
    }

    return pos;
}

Eigen::MatrixXd MultibodyModel::computeJacobian(const JointState& state, int chain_id) const {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, dof_);

    // Simplified Jacobian computation
    for (int i = 0; i <= chain_id && i < dof_ && i < (int)segments_.size(); ++i) {
        double q = state.positions(i);
        J(0, i) = -segments_[i].length * std::sin(q);
        J(1, i) = segments_[i].length * std::cos(q);
    }

    return J;
}

void MultibodyModel::updateSegmentMass(int segment_id, double mass) {
    if (segment_id >= 0 && segment_id < (int)segments_.size()) {
        segments_[segment_id].mass = mass;
    }
}

void MultibodyModel::updateBarMass(double mass) {
    bar_mass_ = mass;
}

Eigen::Matrix3d MultibodyModel::skewSymmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d m;
    m <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return m;
}

} // namespace weightlifting
