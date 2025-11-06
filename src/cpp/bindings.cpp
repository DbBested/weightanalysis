#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "dynamics/multibody_model.h"
#include "dynamics/inverse_dynamics.h"
#include "dynamics/parameter_estimator.h"
#include "physics/bar_bending.h"

namespace py = pybind11;
using namespace weightlifting;

PYBIND11_MODULE(weightanalysis_cpp, m) {
    m.doc() = "High-performance C++ backend for weightlifting analysis";

    // BodySegment
    py::class_<BodySegment>(m, "BodySegment")
        .def(py::init<>())
        .def_readwrite("name", &BodySegment::name)
        .def_readwrite("mass", &BodySegment::mass)
        .def_readwrite("length", &BodySegment::length)
        .def_readwrite("com_offset", &BodySegment::com_offset)
        .def_readwrite("inertia", &BodySegment::inertia);

    // JointState
    py::class_<JointState>(m, "JointState")
        .def(py::init<int>(), py::arg("dof") = 0)
        .def_readwrite("positions", &JointState::positions)
        .def_readwrite("velocities", &JointState::velocities)
        .def_readwrite("accelerations", &JointState::accelerations)
        .def_readwrite("timestamp", &JointState::timestamp);

    // BarState
    py::class_<BarState>(m, "BarState")
        .def(py::init<>())
        .def_readwrite("position", &BarState::position)
        .def_readwrite("velocity", &BarState::velocity)
        .def_readwrite("acceleration", &BarState::acceleration)
        .def_readwrite("orientation", &BarState::orientation)
        .def_readwrite("bending_moment", &BarState::bending_moment)
        .def_readwrite("deflection", &BarState::deflection);

    // InverseDynamicsResult
    py::class_<InverseDynamicsResult>(m, "InverseDynamicsResult")
        .def(py::init<>())
        .def_readwrite("joint_torques", &InverseDynamicsResult::joint_torques)
        .def_readwrite("bar_forces", &InverseDynamicsResult::bar_forces)
        .def_readwrite("reaction_forces", &InverseDynamicsResult::reaction_forces)
        .def_readwrite("timestamps", &InverseDynamicsResult::timestamps)
        .def_readwrite("peak_force", &InverseDynamicsResult::peak_force)
        .def_readwrite("peak_time", &InverseDynamicsResult::peak_time)
        .def_readwrite("power", &InverseDynamicsResult::power);

    // MultibodyModel
    py::class_<MultibodyModel>(m, "MultibodyModel")
        .def(py::init<>())
        .def("initialize_squat_model", &MultibodyModel::initializeSquatModel)
        .def("initialize_bench_model", &MultibodyModel::initializeBenchModel)
        .def("initialize_deadlift_model", &MultibodyModel::initializeDeadliftModel)
        .def("add_segment", &MultibodyModel::addSegment)
        .def("compute_mass_matrix", &MultibodyModel::computeMassMatrix)
        .def("compute_coriolis_forces", &MultibodyModel::computeCoriolisForces)
        .def("compute_gravity_forces", &MultibodyModel::computeGravityForces)
        .def("compute_end_effector_position", &MultibodyModel::computeEndEffectorPosition)
        .def("compute_jacobian", &MultibodyModel::computeJacobian)
        .def("get_degrees_of_freedom", &MultibodyModel::getDegreesOfFreedom)
        .def("get_segments", &MultibodyModel::getSegments)
        .def("update_segment_mass", &MultibodyModel::updateSegmentMass)
        .def("update_bar_mass", &MultibodyModel::updateBarMass);

    // InverseDynamicsSolver
    py::class_<InverseDynamicsSolver>(m, "InverseDynamicsSolver")
        .def(py::init<const MultibodyModel&>())
        .def("solve", &InverseDynamicsSolver::solve)
        .def("compute_inverse_dynamics_rne", &InverseDynamicsSolver::computeInverseDynamicsRNE)
        .def("compute_ground_reaction_force", &InverseDynamicsSolver::computeGroundReactionForce)
        .def("compute_bar_force", &InverseDynamicsSolver::computeBarForce)
        .def("compute_power", &InverseDynamicsSolver::computePower)
        .def("set_external_force", &InverseDynamicsSolver::setExternalForce);

    // ParameterEstimator
    py::class_<ParameterEstimator>(m, "ParameterEstimator")
        .def(py::init<>())
        .def("estimate_bar_mass", &ParameterEstimator::estimateBarMass)
        .def("estimate_segment_masses", &ParameterEstimator::estimateSegmentMasses)
        .def("refine_state_estimates", &ParameterEstimator::refineStateEstimates,
             py::arg("noisy_states"),
             py::arg("model"),
             py::arg("process_noise") = 0.01,
             py::arg("measurement_noise") = 0.1);

    // BarBendingModel
    py::class_<BarBendingModel>(m, "BarBendingModel")
        .def(py::init<>())
        .def("set_bar_parameters", &BarBendingModel::setBarParameters)
        .def("compute_deflection", &BarBendingModel::computeDeflection,
             py::arg("force_positions"),
             py::arg("force_magnitudes"),
             py::arg("num_points") = 50)
        .def("compute_bending_moment", &BarBendingModel::computeBendingMoment,
             py::arg("force_positions"),
             py::arg("force_magnitudes"),
             py::arg("num_points") = 50)
        .def("find_max_deflection", &BarBendingModel::findMaxDeflection)
        .def("compute_strain_energy", &BarBendingModel::computeStrainEnergy)
        .def("get_critical_regions", &BarBendingModel::getCriticalRegions,
             py::arg("bending_moment"),
             py::arg("threshold_fraction") = 0.7)
        .def("get_length", &BarBendingModel::getLength)
        .def("get_stiffness", &BarBendingModel::getStiffness);
}
