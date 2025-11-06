#!/usr/bin/env python3
"""Main application for weightlifting video analysis."""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Import analysis modules
from video_processor import VideoProcessor
from pose_detector import PoseDetector
from bar_tracker import BarTracker, BarDetection
from phase_detector import PhaseDetector
from visualizer import LiftVisualizer
from viewer import InteractiveViewer

# Import C++ backend
try:
    import weightanalysis_cpp as wa_cpp
except ImportError:
    print("Warning: C++ backend not compiled. Please run: mkdir build && cd build && cmake .. && make")
    print("Continuing with limited functionality...")
    wa_cpp = None


def convert_pose_to_joint_state(pose_data, lift_type: str) -> 'wa_cpp.JointState':
    """Convert pose data to joint state for dynamics."""
    if wa_cpp is None:
        return None

    # Simplified conversion - in production, this would use proper IK
    dof = 12  # Default DOF
    state = wa_cpp.JointState(dof)

    # Extract joint angles from pose
    angles = pose_data.get_joint_angles(pose_data, lift_type)

    # Populate state (simplified)
    for i, (joint_name, angle) in enumerate(angles.items()):
        if i < dof:
            state.positions[i] = angle

    state.timestamp = pose_data.timestamp

    return state


def convert_bar_detection_to_state(detection: BarDetection, velocity: np.ndarray,
                                  acceleration: np.ndarray, timestamp: float) -> 'wa_cpp.BarState':
    """Convert bar detection to bar state."""
    if wa_cpp is None:
        return None

    state = wa_cpp.BarState()
    state.position = np.array([detection.center[0], detection.center[1], 0.0])
    state.velocity = velocity
    state.acceleration = acceleration

    return state


def analyze_lift(video_path: str, lift_type: str = 'squat', output_dir: str = 'output'):
    """
    Main analysis pipeline.

    Args:
        video_path: Path to video file
        lift_type: Type of lift (squat, bench, deadlift)
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {lift_type} video: {video_path}")
    print("=" * 60)

    # Step 1: Load and process video
    print("\n[1/7] Loading video...")
    with VideoProcessor(video_path) as video:
        fps = video.fps
        print(f"  Video: {video.width}x{video.height} @ {fps:.1f} fps")
        print(f"  Duration: {video.duration:.2f}s ({video.frame_count} frames)")

        # Downsample if needed
        target_fps = min(fps, 30.0)
        frames_data = video.downsample_frames(target_fps)
        print(f"  Downsampled to {target_fps:.1f} fps ({len(frames_data)} frames)")

    frames = [f for _, _, f in frames_data]
    timestamps = [t for _, t, _ in frames_data]

    # Step 2: Detect lifter pose
    print("\n[2/7] Detecting lifter pose...")
    pose_detector = PoseDetector(model_complexity=2)
    poses = pose_detector.detect_pose_sequence(frames_data)
    print(f"  Detected pose in {len(poses)}/{len(frames)} frames")

    if len(poses) < len(frames) * 0.5:
        print("  Warning: Low pose detection rate. Consider better lighting or camera angle.")

    # Step 3: Track barbell
    print("\n[3/7] Tracking barbell...")
    bar_tracker = BarTracker(initial_detection_method='hough')
    bar_detections = bar_tracker.track_sequence(frames)

    valid_detections = [d for d in bar_detections if d is not None]
    print(f"  Tracked bar in {len(valid_detections)}/{len(frames)} frames")

    # Compute bar kinematics
    bar_positions, bar_velocities, bar_accelerations = bar_tracker.compute_bar_trajectory(
        bar_detections, timestamps, smooth=True
    )
    print(f"  Computed bar trajectory: {len(bar_positions)} points")

    # Step 4: Detect lift phases
    print("\n[4/7] Detecting lift phases...")
    phase_detector = PhaseDetector(lift_type=lift_type)
    phases = phase_detector.detect_phases(bar_positions, bar_velocities, timestamps)

    print(f"  Detected {len(phases)} phases:")
    for phase in phases:
        print(f"    - {phase.name}: {phase.duration:.2f}s")

    # Step 5: Run inverse dynamics (if C++ backend available)
    print("\n[5/7] Computing inverse dynamics...")

    if wa_cpp is not None:
        # Initialize multibody model
        model = wa_cpp.MultibodyModel()
        if lift_type == 'squat':
            model.initialize_squat_model()
        elif lift_type == 'bench':
            model.initialize_bench_model()
        elif lift_type == 'deadlift':
            model.initialize_deadlift_model()

        print(f"  Initialized {lift_type} model with {model.get_degrees_of_freedom()} DOF")

        # Convert tracking data to states
        joint_states = []
        bar_states = []

        for i in range(min(len(poses), len(bar_positions))):
            # Joint state (simplified - would need proper IK)
            js = wa_cpp.JointState(model.get_degrees_of_freedom())
            js.timestamp = poses[i].timestamp if i < len(poses) else timestamps[i]

            # Bar state
            bs = wa_cpp.BarState()
            bs.position = bar_positions[i]
            bs.velocity = bar_velocities[i]
            bs.acceleration = bar_accelerations[i]

            joint_states.append(js)
            bar_states.append(bs)

        # Solve inverse dynamics
        solver = wa_cpp.InverseDynamicsSolver(model)
        dynamics_result = solver.solve(joint_states, bar_states)

        print(f"  Peak force: {dynamics_result.peak_force:.0f} N at t={dynamics_result.peak_time:.2f}s")
        print(f"  Peak power: {max(dynamics_result.power):.0f} W")

        # Estimate bar mass
        estimator = wa_cpp.ParameterEstimator()
        estimated_mass = estimator.estimate_bar_mass(bar_states, dynamics_result.bar_forces)
        print(f"  Estimated bar mass: {estimated_mass:.1f} kg")

        # Compute bar bending
        bending_model = wa_cpp.BarBendingModel()
        bending_model.set_bar_parameters(2.2, 0.028, 200e9)  # Standard barbell

        # Assume forces at hands (0.3 and 0.7 normalized positions)
        force_positions = [0.3, 0.7]
        bar_forces_list = []

        for bf in dynamics_result.bar_forces:
            force_mag = np.linalg.norm(bf)
            bar_forces_list.append(force_mag / 2)  # Split between hands

    else:
        print("  Skipping dynamics (C++ backend not available)")
        dynamics_result = None
        bending_model = None

    # Step 6: Create visualizations
    print("\n[6/7] Creating visualizations...")
    visualizer = LiftVisualizer(frames[0].shape[1], frames[0].shape[0])

    annotated_frames = []

    for i in tqdm(range(len(frames)), desc="  Rendering"):
        frame = frames[i]

        # Get current data
        current_phase = phase_detector.get_phase_at_frame(phases, i)
        current_time = timestamps[i]

        bar_detection = bar_detections[i] if i < len(bar_detections) else None

        if bar_detection and dynamics_result and i < len(dynamics_result.bar_forces):
            # Get force
            force_vector = dynamics_result.bar_forces[i]

            # Get bending
            if bending_model:
                force_mag = np.linalg.norm(force_vector) / 2
                deflection = bending_model.compute_deflection(force_positions, [force_mag, force_mag])
                bending_moment = bending_model.compute_bending_moment(force_positions, [force_mag, force_mag])
                critical_regions = bending_model.get_critical_regions(bending_moment)
            else:
                deflection = []
                critical_regions = []

            # Trajectory
            trajectory = [(p[0], p[1]) for p in bar_positions[:i+1]]

            # Summary
            summary = {
                'Peak Force': f"{dynamics_result.peak_force:.0f}N",
                'Curr Force': f"{np.linalg.norm(force_vector):.0f}N",
                'Phase': current_phase,
            }

            # Create complete visualization
            annotated = visualizer.create_complete_visualization(
                frame, bar_detection.center, bar_detection.endpoints,
                force_vector, deflection, critical_regions,
                trajectory, i, current_phase, current_time, summary
            )
        else:
            # Basic annotation
            annotated = visualizer.draw_phase_label(frame, current_phase, current_time)

        annotated_frames.append(annotated)

    # Step 7: Interactive viewer
    print("\n[7/7] Launching interactive viewer...")
    viewer = InteractiveViewer("Weightlifting Analysis")
    viewer.load_frames(annotated_frames, target_fps)

    print("\nViewer controls:")
    print("  SPACE: Play/Pause")
    print("  LEFT/RIGHT: Navigate frames")
    print("  +/-: Adjust speed")
    print("  Q/ESC: Quit")
    print("\nStarting viewer...")

    viewer.run()

    # Save results
    print("\nSaving results...")
    output_video = output_path / f"analysis_{Path(video_path).stem}.mp4"
    viewer.export_video(str(output_video), target_fps)
    print(f"  Saved video: {output_video}")

    # Save summary JSON
    summary_data = {
        'video': video_path,
        'lift_type': lift_type,
        'duration': timestamps[-1] if timestamps else 0,
        'phases': [
            {
                'name': p.name,
                'start_time': p.start_time,
                'end_time': p.end_time,
                'duration': p.duration
            }
            for p in phases
        ],
    }

    if dynamics_result:
        summary_data['dynamics'] = {
            'peak_force_N': float(dynamics_result.peak_force),
            'peak_force_time_s': float(dynamics_result.peak_time),
            'peak_power_W': float(max(dynamics_result.power)),
            'estimated_mass_kg': float(estimated_mass) if 'estimated_mass' in locals() else None,
        }

    summary_json = output_path / f"summary_{Path(video_path).stem}.json"
    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved summary: {summary_json}")

    print("\nAnalysis complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze weightlifting videos with biomechanical insights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video squat.mp4 --lift-type squat
  python main.py --video bench.mp4 --lift-type bench --output results/
        """
    )

    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--lift-type', default='squat',
                       choices=['squat', 'bench', 'deadlift'],
                       help='Type of lift')
    parser.add_argument('--output', default='output',
                       help='Output directory')

    args = parser.parse_args()

    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    try:
        analyze_lift(args.video, args.lift_type, args.output)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
