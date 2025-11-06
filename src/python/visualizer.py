"""Visualization utilities for overlaying analysis on video frames."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class LiftVisualizer:
    """Visualize lift analysis with force arrows, bending, and trajectories."""

    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize visualizer.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.width = frame_width
        self.height = frame_height
        self.trajectory_history = 30  # Number of frames to show in trajectory

        # Color scheme
        self.colors = {
            'force_arrow': (0, 255, 255),      # Cyan
            'bending': (255, 0, 255),          # Magenta
            'trajectory': (0, 255, 0),         # Green
            'phase_setup': (200, 200, 200),    # Gray
            'phase_descent': (0, 165, 255),    # Orange
            'phase_bottom': (0, 0, 255),       # Red
            'phase_ascent': (0, 255, 0),       # Green
            'phase_completion': (255, 255, 0), # Yellow
        }

    def draw_force_arrow(self, frame: np.ndarray,
                        position: Tuple[float, float],
                        force_vector: np.ndarray,
                        scale: float = 0.05) -> np.ndarray:
        """
        Draw force arrow at position.

        Args:
            frame: Input frame
            position: (x, y) position
            force_vector: 3D force vector [Fx, Fy, Fz] in Newtons
            scale: Scaling factor for arrow length

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        x, y = int(position[0]), int(position[1])

        # Project force to 2D (use x and y components)
        fx, fy = force_vector[0], force_vector[1]
        force_mag = np.linalg.norm(force_vector)

        # Arrow endpoint
        arrow_length = force_mag * scale
        end_x = int(x + fx * scale)
        end_y = int(y + fy * scale)

        # Draw arrow
        cv2.arrowedLine(annotated, (x, y), (end_x, end_y),
                       self.colors['force_arrow'], 3, tipLength=0.3)

        # Draw force magnitude text
        force_text = f"{force_mag:.0f}N"
        text_pos = (end_x + 10, end_y)

        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(force_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated,
                     (text_pos[0] - 5, text_pos[1] - text_h - 5),
                     (text_pos[0] + text_w + 5, text_pos[1] + 5),
                     (0, 0, 0), -1)

        cv2.putText(annotated, force_text, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def draw_bar_bending(self, frame: np.ndarray,
                        bar_center: Tuple[float, float],
                        bar_endpoints: Tuple[Tuple[float, float], Tuple[float, float]],
                        deflection: List[float],
                        critical_regions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw bar bending visualization.

        Args:
            frame: Input frame
            bar_center: Bar center position
            bar_endpoints: Bar endpoints
            deflection: Deflection values along bar
            critical_regions: List of (start, end) indices for critical bending

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        if not deflection or len(deflection) < 2:
            return annotated

        (x1, y1), (x2, y2) = bar_endpoints

        # Draw nominal bar position (straight line)
        cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                (128, 128, 128), 1, cv2.LINE_AA)

        # Draw deflected bar shape
        num_points = len(deflection)
        bar_points = []

        for i in range(num_points):
            t = i / (num_points - 1)  # Normalized position along bar

            # Interpolate position along bar
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # Add deflection (perpendicular to bar)
            bar_angle = np.arctan2(y2 - y1, x2 - x1)
            perp_angle = bar_angle + np.pi / 2

            # Scale deflection for visibility
            defl_scale = 100.0  # Adjust for visualization
            x_defl = x + deflection[i] * defl_scale * np.cos(perp_angle)
            y_defl = y + deflection[i] * defl_scale * np.sin(perp_angle)

            bar_points.append((int(x_defl), int(y_defl)))

        # Draw deflected bar
        if len(bar_points) > 1:
            for i in range(len(bar_points) - 1):
                cv2.line(annotated, bar_points[i], bar_points[i+1],
                        self.colors['bending'], 2, cv2.LINE_AA)

        # Highlight critical bending regions
        for start_idx, end_idx in critical_regions:
            if 0 <= start_idx < len(bar_points) and 0 <= end_idx < len(bar_points):
                for i in range(start_idx, min(end_idx, len(bar_points) - 1)):
                    cv2.line(annotated, bar_points[i], bar_points[i+1],
                            (0, 0, 255), 4, cv2.LINE_AA)

        # Add bending annotation
        max_defl = max(abs(d) for d in deflection) if deflection else 0
        defl_text = f"Max bend: {max_defl*1000:.1f}mm"
        cv2.putText(annotated, defl_text, (int(bar_center[0]) - 50, int(bar_center[1]) - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def draw_trajectory(self, frame: np.ndarray,
                       positions: List[Tuple[float, float]],
                       current_idx: int) -> np.ndarray:
        """
        Draw trajectory trail.

        Args:
            frame: Input frame
            positions: List of (x, y) positions
            current_idx: Current frame index

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Determine range of trajectory to show
        start_idx = max(0, current_idx - self.trajectory_history)
        end_idx = min(len(positions), current_idx + 1)

        trail_positions = positions[start_idx:end_idx]

        # Draw trajectory with fading effect
        for i in range(len(trail_positions) - 1):
            # Fade factor (newer points are brighter)
            fade = (i + 1) / len(trail_positions)
            color = tuple(int(c * fade) for c in self.colors['trajectory'])

            pt1 = (int(trail_positions[i][0]), int(trail_positions[i][1]))
            pt2 = (int(trail_positions[i+1][0]), int(trail_positions[i+1][1]))

            cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw current position
        if trail_positions:
            curr_pt = (int(trail_positions[-1][0]), int(trail_positions[-1][1]))
            cv2.circle(annotated, curr_pt, 5, self.colors['trajectory'], -1)

        return annotated

    def draw_phase_label(self, frame: np.ndarray, phase_name: str, time: float) -> np.ndarray:
        """
        Draw phase label and timing.

        Args:
            frame: Input frame
            phase_name: Name of current phase
            time: Current time in seconds

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Get phase color
        color_key = f'phase_{phase_name}'
        color = self.colors.get(color_key, (255, 255, 255))

        # Create label
        label = f"{phase_name.upper()} | {time:.2f}s"

        # Position at top of frame
        text_pos = (20, 40)

        # Background rectangle
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(annotated,
                     (text_pos[0] - 10, text_pos[1] - text_h - 10),
                     (text_pos[0] + text_w + 10, text_pos[1] + 10),
                     color, -1)

        # Border
        cv2.rectangle(annotated,
                     (text_pos[0] - 10, text_pos[1] - text_h - 10),
                     (text_pos[0] + text_w + 10, text_pos[1] + 10),
                     (255, 255, 255), 2)

        # Text
        cv2.putText(annotated, label, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return annotated

    def draw_summary_overlay(self, frame: np.ndarray, summary: dict) -> np.ndarray:
        """
        Draw summary statistics overlay.

        Args:
            frame: Input frame
            summary: Dictionary with summary stats

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Position at bottom-right
        x_start = self.width - 300
        y_start = self.height - 150

        # Background
        cv2.rectangle(annotated,
                     (x_start - 10, y_start - 10),
                     (self.width - 10, self.height - 10),
                     (0, 0, 0), -1)

        cv2.rectangle(annotated,
                     (x_start - 10, y_start - 10),
                     (self.width - 10, self.height - 10),
                     (255, 255, 255), 2)

        # Draw summary lines
        y_offset = y_start
        line_height = 25

        for key, value in summary.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"

            cv2.putText(annotated, text, (x_start, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height

        return annotated

    def create_complete_visualization(self, frame: np.ndarray,
                                     bar_center: Tuple[float, float],
                                     bar_endpoints: Tuple[Tuple[float, float], Tuple[float, float]],
                                     force_vector: np.ndarray,
                                     deflection: List[float],
                                     critical_regions: List[Tuple[int, int]],
                                     trajectory: List[Tuple[float, float]],
                                     current_idx: int,
                                     phase_name: str,
                                     time: float,
                                     summary: Optional[dict] = None) -> np.ndarray:
        """
        Create complete visualization with all overlays.

        Args:
            frame: Input frame
            bar_center: Bar center position
            bar_endpoints: Bar endpoints
            force_vector: Force vector
            deflection: Deflection values
            critical_regions: Critical bending regions
            trajectory: Bar trajectory
            current_idx: Current frame index
            phase_name: Current phase
            time: Current time
            summary: Optional summary statistics

        Returns:
            Fully annotated frame
        """
        annotated = frame.copy()

        # Draw in order (background to foreground)
        annotated = self.draw_trajectory(annotated, trajectory, current_idx)
        annotated = self.draw_bar_bending(annotated, bar_center, bar_endpoints,
                                         deflection, critical_regions)
        annotated = self.draw_force_arrow(annotated, bar_center, force_vector)
        annotated = self.draw_phase_label(annotated, phase_name, time)

        if summary:
            annotated = self.draw_summary_overlay(annotated, summary)

        return annotated
