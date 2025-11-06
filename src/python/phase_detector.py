"""Phase detection for identifying lift phases (descent, bottom, ascent)."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks


@dataclass
class LiftPhase:
    """Represents a phase of the lift."""
    name: str  # 'setup', 'descent', 'bottom', 'ascent', 'completion'
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float


class PhaseDetector:
    """Detect and label phases of weightlifting movements."""

    PHASE_NAMES = ['setup', 'descent', 'bottom', 'ascent', 'completion']

    def __init__(self, lift_type: str = 'squat'):
        """
        Initialize phase detector.

        Args:
            lift_type: Type of lift (squat, bench, deadlift, etc.)
        """
        self.lift_type = lift_type

    def detect_phases(self, bar_positions: np.ndarray,
                     bar_velocities: np.ndarray,
                     timestamps: List[float]) -> List[LiftPhase]:
        """
        Detect phases based on bar kinematics.

        Args:
            bar_positions: Nx3 array of bar positions
            bar_velocities: Nx3 array of bar velocities
            timestamps: List of timestamps

        Returns:
            List of LiftPhase objects
        """
        if len(bar_positions) < 10:
            return []

        # Use vertical position and velocity for phase detection
        y_pos = bar_positions[:, 1]  # Vertical position
        y_vel = bar_velocities[:, 1]  # Vertical velocity

        # Find key events
        phases = []

        # 1. Setup phase: minimal movement at start
        setup_end = self._find_movement_start(y_vel)
        if setup_end > 0:
            phases.append(LiftPhase(
                name='setup',
                start_frame=0,
                end_frame=setup_end,
                start_time=timestamps[0],
                end_time=timestamps[setup_end],
                duration=timestamps[setup_end] - timestamps[0]
            ))

        # 2. Descent phase: bar moving down (positive y_vel in image coords)
        descent_start = setup_end
        descent_end = self._find_bottom_position(y_pos, descent_start)

        if descent_end > descent_start:
            phases.append(LiftPhase(
                name='descent',
                start_frame=descent_start,
                end_frame=descent_end,
                start_time=timestamps[descent_start],
                end_time=timestamps[descent_end],
                duration=timestamps[descent_end] - timestamps[descent_start]
            ))

        # 3. Bottom phase: transition zone (low velocity)
        bottom_start = descent_end
        bottom_end = self._find_ascent_start(y_vel, bottom_start)

        if bottom_end > bottom_start:
            phases.append(LiftPhase(
                name='bottom',
                start_frame=bottom_start,
                end_frame=bottom_end,
                start_time=timestamps[bottom_start],
                end_time=timestamps[bottom_end],
                duration=timestamps[bottom_end] - timestamps[bottom_start]
            ))

        # 4. Ascent phase: bar moving up
        ascent_start = bottom_end
        ascent_end = self._find_ascent_end(y_vel, ascent_start)

        if ascent_end > ascent_start:
            phases.append(LiftPhase(
                name='ascent',
                start_frame=ascent_start,
                end_frame=ascent_end,
                start_time=timestamps[ascent_start],
                end_time=timestamps[ascent_end],
                duration=timestamps[ascent_end] - timestamps[ascent_start]
            ))

        # 5. Completion phase: minimal movement at end
        if ascent_end < len(timestamps) - 1:
            phases.append(LiftPhase(
                name='completion',
                start_frame=ascent_end,
                end_frame=len(timestamps) - 1,
                start_time=timestamps[ascent_end],
                end_time=timestamps[-1],
                duration=timestamps[-1] - timestamps[ascent_end]
            ))

        return phases

    def _find_movement_start(self, velocities: np.ndarray, threshold: float = 2.0) -> int:
        """Find when significant movement starts."""
        for i in range(len(velocities)):
            if np.abs(velocities[i]) > threshold:
                return max(0, i - 2)  # Back up slightly
        return 0

    def _find_bottom_position(self, positions: np.ndarray, start_idx: int) -> int:
        """Find the lowest point (highest y-value in image coordinates)."""
        if start_idx >= len(positions) - 1:
            return start_idx

        # Find local maxima in position (lowest point)
        search_region = positions[start_idx:]
        peaks, _ = find_peaks(search_region, distance=10)

        if len(peaks) > 0:
            return start_idx + peaks[0]

        # Fallback: find maximum value
        max_idx = np.argmax(search_region)
        return start_idx + max_idx

    def _find_ascent_start(self, velocities: np.ndarray, start_idx: int, threshold: float = -2.0) -> int:
        """Find when upward movement starts (negative velocity in image coords)."""
        for i in range(start_idx, len(velocities)):
            if velocities[i] < threshold:
                return i
        return min(start_idx + 5, len(velocities) - 1)

    def _find_ascent_end(self, velocities: np.ndarray, start_idx: int, threshold: float = 1.0) -> int:
        """Find when upward movement ends."""
        for i in range(start_idx, len(velocities)):
            if np.abs(velocities[i]) < threshold:
                return i
        return len(velocities) - 1

    def analyze_phase_timing(self, phases: List[LiftPhase]) -> dict:
        """
        Analyze timing characteristics of phases.

        Args:
            phases: List of LiftPhase objects

        Returns:
            Dictionary with timing analysis
        """
        analysis = {}

        phase_dict = {p.name: p for p in phases}

        # Total lift time
        if phases:
            total_time = phases[-1].end_time - phases[0].start_time
            analysis['total_time'] = total_time

        # Individual phase durations
        for phase in phases:
            analysis[f'{phase.name}_duration'] = phase.duration

        # Descent to ascent ratio
        if 'descent' in phase_dict and 'ascent' in phase_dict:
            descent_time = phase_dict['descent'].duration
            ascent_time = phase_dict['ascent'].duration
            if ascent_time > 0:
                analysis['descent_to_ascent_ratio'] = descent_time / ascent_time

        # Time to bottom
        if 'descent' in phase_dict:
            analysis['time_to_bottom'] = phase_dict['descent'].end_time - phases[0].start_time

        return analysis

    def get_phase_at_frame(self, phases: List[LiftPhase], frame_idx: int) -> str:
        """
        Get the phase name at a specific frame.

        Args:
            phases: List of LiftPhase objects
            frame_idx: Frame index

        Returns:
            Phase name or 'unknown'
        """
        for phase in phases:
            if phase.start_frame <= frame_idx <= phase.end_frame:
                return phase.name
        return 'unknown'

    def detect_sticking_point(self, bar_velocities: np.ndarray,
                             phases: List[LiftPhase]) -> Tuple[int, float]:
        """
        Detect sticking point (slowest point during ascent).

        Args:
            bar_velocities: Nx3 array of velocities
            phases: List of LiftPhase objects

        Returns:
            Tuple of (frame_idx, velocity_magnitude)
        """
        # Find ascent phase
        ascent_phase = None
        for phase in phases:
            if phase.name == 'ascent':
                ascent_phase = phase
                break

        if not ascent_phase:
            return -1, 0.0

        # Find slowest point during ascent
        start, end = ascent_phase.start_frame, ascent_phase.end_frame
        ascent_velocities = bar_velocities[start:end+1]

        # Compute velocity magnitudes
        vel_magnitudes = np.linalg.norm(ascent_velocities, axis=1)

        # Find minimum (sticking point)
        min_idx = np.argmin(vel_magnitudes)
        sticking_frame = start + min_idx
        sticking_velocity = vel_magnitudes[min_idx]

        return sticking_frame, sticking_velocity
