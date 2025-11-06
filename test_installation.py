#!/usr/bin/env python3
"""Quick installation verification script."""

import sys
sys.path.insert(0, 'src/python')

def test_python_modules():
    """Test Python dependencies."""
    print("Testing Python modules...")
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV failed: {e}")
        return False

    try:
        import mediapipe as mp
        print(f"  ✓ MediaPipe {mp.__version__}")
    except ImportError as e:
        print(f"  ✗ MediaPipe failed: {e}")
        return False

    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy failed: {e}")
        return False

    try:
        import scipy
        print(f"  ✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"  ✗ SciPy failed: {e}")
        return False

    return True


def test_cpp_backend():
    """Test C++ backend."""
    print("\nTesting C++ backend...")
    try:
        import weightanalysis_cpp as wa
        print("  ✓ C++ module imported")

        # Test MultibodyModel
        model = wa.MultibodyModel()
        model.initialize_squat_model()
        print(f"  ✓ MultibodyModel (DOF: {model.get_degrees_of_freedom()})")

        # Test InverseDynamicsSolver
        solver = wa.InverseDynamicsSolver(model)
        print("  ✓ InverseDynamicsSolver")

        # Test ParameterEstimator
        estimator = wa.ParameterEstimator()
        print("  ✓ ParameterEstimator")

        # Test BarBendingModel
        bending = wa.BarBendingModel()
        print(f"  ✓ BarBendingModel (length: {bending.get_length():.2f}m)")

        return True
    except ImportError as e:
        print(f"  ✗ C++ backend import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ C++ backend test failed: {e}")
        return False


def test_python_modules_functionality():
    """Test Python analysis modules."""
    print("\nTesting Python analysis modules...")
    try:
        from video_processor import VideoProcessor
        print("  ✓ VideoProcessor")

        from pose_detector import PoseDetector
        print("  ✓ PoseDetector")

        from bar_tracker import BarTracker
        print("  ✓ BarTracker")

        from phase_detector import PhaseDetector
        print("  ✓ PhaseDetector")

        from visualizer import LiftVisualizer
        print("  ✓ LiftVisualizer")

        from viewer import InteractiveViewer
        print("  ✓ InteractiveViewer")

        return True
    except ImportError as e:
        print(f"  ✗ Module import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Weightlifting Analysis - Installation Verification")
    print("=" * 60)

    all_passed = True

    # Test Python modules
    if not test_python_modules():
        all_passed = False

    # Test C++ backend
    if not test_cpp_backend():
        all_passed = False

    # Test Python analysis modules
    if not test_python_modules_functionality():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All components installed correctly!")
        print("=" * 60)
        print("\nYou can now analyze videos:")
        print("  python3 src/python/main.py --video VIDEO.mp4 --lift-type squat")
        return 0
    else:
        print("FAILURE: Some components failed to install")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
