"""Interactive viewer for weightlifting analysis results."""

import cv2
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass


@dataclass
class ViewerState:
    """State of the interactive viewer."""
    current_frame: int = 0
    playing: bool = False
    playback_speed: float = 1.0
    show_trajectory: bool = True
    show_forces: bool = True
    show_bending: bool = True
    show_phase: bool = True
    show_summary: bool = True


class InteractiveViewer:
    """Interactive OpenCV-based viewer for analysis results."""

    def __init__(self, window_name: str = "Weightlifting Analysis"):
        """
        Initialize viewer.

        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.state = ViewerState()

        # Keyboard controls
        self.key_bindings = {
            ord(' '): self.toggle_play,
            ord('r'): self.reset,
            ord('t'): self.toggle_trajectory,
            ord('f'): self.toggle_forces,
            ord('b'): self.toggle_bending,
            ord('p'): self.toggle_phase,
            ord('s'): self.toggle_summary,
            ord('+'): self.increase_speed,
            ord('='): self.increase_speed,
            ord('-'): self.decrease_speed,
            ord('q'): lambda: 'quit',
            27: lambda: 'quit',  # ESC
        }

        # Frame data
        self.frames = []
        self.total_frames = 0
        self.fps = 30.0

    def load_frames(self, frames: List[np.ndarray], fps: float = 30.0):
        """
        Load frames for display.

        Args:
            frames: List of annotated frames
            fps: Frames per second
        """
        self.frames = frames
        self.total_frames = len(frames)
        self.fps = fps
        self.state.current_frame = 0

    def run(self):
        """
        Run the interactive viewer loop.

        Returns:
            Final frame index when viewer is closed
        """
        if not self.frames:
            print("No frames loaded!")
            return 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        # Create trackbar for frame navigation
        cv2.createTrackbar('Frame', self.window_name, 0, self.total_frames - 1,
                          self._on_trackbar)

        delay = int(1000 / (self.fps * self.state.playback_speed))

        while True:
            # Get current frame
            if 0 <= self.state.current_frame < self.total_frames:
                frame = self.frames[self.state.current_frame].copy()

                # Add control instructions
                frame = self._draw_controls(frame)

                # Update trackbar
                cv2.setTrackbarPos('Frame', self.window_name, self.state.current_frame)

                # Display
                cv2.imshow(self.window_name, frame)

            # Handle playback
            if self.state.playing:
                self.state.current_frame += 1
                if self.state.current_frame >= self.total_frames:
                    self.state.current_frame = 0  # Loop

            # Process keyboard input
            key = cv2.waitKey(delay if self.state.playing else 50) & 0xFF

            if key == 255:  # No key pressed
                continue

            # Handle key bindings
            if key in self.key_bindings:
                result = self.key_bindings[key]()
                if result == 'quit':
                    break

            # Arrow keys for frame navigation
            elif key == 81 or key == 2:  # Left arrow
                self.state.current_frame = max(0, self.state.current_frame - 1)
                self.state.playing = False
            elif key == 83 or key == 3:  # Right arrow
                self.state.current_frame = min(self.total_frames - 1, self.state.current_frame + 1)
                self.state.playing = False

        cv2.destroyWindow(self.window_name)
        return self.state.current_frame

    def _on_trackbar(self, value: int):
        """Trackbar callback."""
        self.state.current_frame = value
        self.state.playing = False

    def _draw_controls(self, frame: np.ndarray) -> np.ndarray:
        """Draw control instructions on frame."""
        annotated = frame.copy()

        controls = [
            "SPACE: Play/Pause",
            "LEFT/RIGHT: Step frame",
            "+/-: Speed",
            "T: Trajectory",
            "F: Forces",
            "B: Bending",
            "P: Phase",
            "S: Summary",
            "R: Reset",
            "Q/ESC: Quit"
        ]

        y_offset = frame.shape[0] - len(controls) * 20 - 20
        x_offset = 10

        # Semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay,
                     (x_offset - 5, y_offset - 15),
                     (x_offset + 200, frame.shape[0] - 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        # Draw controls
        for i, control in enumerate(controls):
            y_pos = y_offset + i * 20
            cv2.putText(annotated, control, (x_offset, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Playback info
        info = f"Frame: {self.state.current_frame + 1}/{self.total_frames} | "
        info += f"Speed: {self.state.playback_speed:.1f}x | "
        info += f"{'PLAYING' if self.state.playing else 'PAUSED'}"

        cv2.rectangle(annotated, (0, 0), (len(info) * 10, 35), (0, 0, 0), -1)
        cv2.putText(annotated, info, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated

    # Control methods
    def toggle_play(self):
        """Toggle play/pause."""
        self.state.playing = not self.state.playing

    def reset(self):
        """Reset to first frame."""
        self.state.current_frame = 0
        self.state.playing = False

    def toggle_trajectory(self):
        """Toggle trajectory display."""
        self.state.show_trajectory = not self.state.show_trajectory

    def toggle_forces(self):
        """Toggle force arrows display."""
        self.state.show_forces = not self.state.show_forces

    def toggle_bending(self):
        """Toggle bending display."""
        self.state.show_bending = not self.state.show_bending

    def toggle_phase(self):
        """Toggle phase labels."""
        self.state.show_phase = not self.state.show_phase

    def toggle_summary(self):
        """Toggle summary overlay."""
        self.state.show_summary = not self.state.show_summary

    def increase_speed(self):
        """Increase playback speed."""
        self.state.playback_speed = min(4.0, self.state.playback_speed * 1.5)

    def decrease_speed(self):
        """Decrease playback speed."""
        self.state.playback_speed = max(0.25, self.state.playback_speed / 1.5)

    def export_video(self, output_path: str, fps: Optional[float] = None):
        """
        Export annotated frames to video file.

        Args:
            output_path: Path to output video
            fps: Output FPS (uses input FPS if None)
        """
        if not self.frames:
            print("No frames to export!")
            return

        output_fps = fps if fps else self.fps

        # Get frame dimensions
        height, width = self.frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        print(f"Exporting video to {output_path}...")

        for i, frame in enumerate(self.frames):
            out.write(frame)
            if (i + 1) % 30 == 0:
                print(f"  Progress: {i+1}/{len(self.frames)} frames")

        out.release()
        print(f"Video exported successfully!")

    def export_frames(self, output_dir: str, prefix: str = "frame"):
        """
        Export individual frames as images.

        Args:
            output_dir: Output directory
            prefix: Filename prefix
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        print(f"Exporting frames to {output_dir}...")

        for i, frame in enumerate(self.frames):
            filename = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            cv2.imwrite(filename, frame)

        print(f"Exported {len(self.frames)} frames!")
