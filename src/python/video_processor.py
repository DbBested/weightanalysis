"""Video processing utilities for extracting and managing frames."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class VideoProcessor:
    """Handles video I/O and frame extraction."""

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def extract_frames(self, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract all frames from the video.

        Args:
            max_frames: Maximum number of frames to extract (None for all)

        Returns:
            List of frames as numpy arrays
        """
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frames.append(frame)
            frame_idx += 1

            if max_frames and frame_idx >= max_frames:
                break

        return frames

    def extract_frames_generator(self):
        """
        Generator for frames to save memory.

        Yields:
            Tuple of (frame_idx, timestamp, frame)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_idx / self.fps if self.fps > 0 else 0
            yield frame_idx, timestamp, frame
            frame_idx += 1

    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            Frame as numpy array or None
        """
        frame_idx = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame_at_index(self, idx: int) -> Optional[np.ndarray]:
        """
        Get frame at specific index.

        Args:
            idx: Frame index

        Returns:
            Frame as numpy array or None
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def downsample_frames(self, target_fps: float = 30.0) -> List[Tuple[int, float, np.ndarray]]:
        """
        Downsample video to target FPS.

        Args:
            target_fps: Target frames per second

        Returns:
            List of (frame_idx, timestamp, frame) tuples
        """
        if self.fps <= target_fps:
            return [(i, t, f) for i, t, f in self.extract_frames_generator()]

        skip_factor = int(self.fps / target_fps)
        downsampled = []

        for frame_idx, timestamp, frame in self.extract_frames_generator():
            if frame_idx % skip_factor == 0:
                downsampled.append((frame_idx, timestamp, frame))

        return downsampled

    def preprocess_frame(self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess frame (resize, normalize).

        Args:
            frame: Input frame
            target_size: Target (width, height) or None to keep original

        Returns:
            Preprocessed frame
        """
        processed = frame.copy()

        if target_size:
            processed = cv2.resize(processed, target_size)

        return processed

    def detect_motion_regions(self, frames: List[np.ndarray], threshold: float = 30.0) -> np.ndarray:
        """
        Detect regions of motion across frames.

        Args:
            frames: List of frames
            threshold: Motion detection threshold

        Returns:
            Binary mask of motion regions
        """
        if len(frames) < 2:
            return np.zeros((self.height, self.width), dtype=np.uint8)

        motion_mask = np.zeros((self.height, self.width), dtype=np.float32)

        for i in range(1, len(frames)):
            diff = cv2.absdiff(
                cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            )
            motion_mask += (diff > threshold).astype(np.float32)

        motion_mask = (motion_mask > 0).astype(np.uint8) * 255
        return motion_mask

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
