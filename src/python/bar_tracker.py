"""Bar detection and tracking using computer vision."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import savgol_filter


@dataclass
class BarDetection:
    """Bar detection result for a single frame."""
    center: Tuple[float, float]  # (x, y) center position
    endpoints: Tuple[Tuple[float, float], Tuple[float, float]]  # Left and right ends
    angle: float  # Rotation angle in radians
    confidence: float  # Detection confidence
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)


class BarTracker:
    """Track barbell position and orientation in video."""

    def __init__(self, initial_detection_method: str = 'hough'):
        """
        Initialize bar tracker.

        Args:
            initial_detection_method: 'hough' for Hough line detection or 'template' for template matching
        """
        self.method = initial_detection_method
        self.prev_detection = None

    def detect_bar_initial(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None,
                          person_mask: Optional[np.ndarray] = None) -> Optional[BarDetection]:
        """
        Detect bar in the first frame (or manually).

        Args:
            frame: Input frame
            roi: Region of interest (x, y, w, h) or None for full frame
            person_mask: Optional person segmentation mask to exclude

        Returns:
            BarDetection or None
        """
        if roi:
            x, y, w, h = roi
            search_region = frame[y:y+h, x:x+w]
            mask_region = person_mask[y:y+h, x:x+w] if person_mask is not None else None
        else:
            search_region = frame
            mask_region = person_mask
            x, y = 0, 0

        if self.method == 'hough':
            detection = self._detect_bar_hough(search_region, mask_region)
        else:
            detection = self._detect_bar_color(search_region)

        if detection and roi:
            # Adjust coordinates back to full frame
            cx, cy = detection.center
            detection.center = (cx + x, cy + y)

            (x1, y1), (x2, y2) = detection.endpoints
            detection.endpoints = ((x1 + x, y1 + y), (x2 + x, y2 + y))

            bx, by, bw, bh = detection.bounding_box
            detection.bounding_box = (bx + x, by + y, bw, bh)

        return detection

    def _detect_bar_hough(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[BarDetection]:
        """Detect bar using Hough line transform.

        Args:
            frame: Input frame
            mask: Optional mask to exclude regions (0 = exclude, 255 = include)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply mask if provided (invert person mask to exclude person)
        if mask is not None:
            # Invert mask so person area is black (excluded)
            inverted_mask = cv2.bitwise_not(mask)
            gray = cv2.bitwise_and(gray, gray, mask=inverted_mask)

        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for better detection

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                               minLineLength=100, maxLineGap=10)

        if lines is None:
            return None

        # Find the most horizontal long line (likely the bar)
        best_line = None
        max_length = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))

            # Prefer horizontal lines (small angle)
            if angle < np.pi / 6 and length > max_length:  # Within 30 degrees
                max_length = length
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        angle = np.arctan2(y2 - y1, x2 - x1)

        # Bounding box
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        bbox = (min_x, min_y - 10, max_x - min_x, max_y - min_y + 20)

        return BarDetection(
            center=center,
            endpoints=((x1, y1), (x2, y2)),
            angle=angle,
            confidence=min(1.0, length / 200),
            bounding_box=bbox
        )

    def _detect_bar_color(self, frame: np.ndarray) -> Optional[BarDetection]:
        """Detect bar using color/intensity (metallic surfaces are bright)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold for bright metallic surfaces
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the longest thin contour
        best_contour = None
        max_aspect_ratio = 0

        for contour in contours:
            if len(contour) < 5:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            if w == 0 or h == 0:
                continue

            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

            # Bar should be very elongated
            if aspect_ratio > 10 and aspect_ratio > max_aspect_ratio:
                max_aspect_ratio = aspect_ratio
                best_contour = contour
                best_rect = rect

        if best_contour is None:
            return None

        (cx, cy), (w, h), angle_deg = best_rect
        angle_rad = np.deg2rad(angle_deg)

        # Compute endpoints
        length = max(w, h)
        dx = length / 2 * np.cos(angle_rad)
        dy = length / 2 * np.sin(angle_rad)

        endpoint1 = (cx - dx, cy - dy)
        endpoint2 = (cx + dx, cy + dy)

        bbox = cv2.boundingRect(best_contour)

        return BarDetection(
            center=(cx, cy),
            endpoints=(endpoint1, endpoint2),
            angle=angle_rad,
            confidence=min(1.0, max_aspect_ratio / 20),
            bounding_box=bbox
        )

    def track_bar(self, frame: np.ndarray, prev_detection: BarDetection,
                  person_mask: Optional[np.ndarray] = None) -> Optional[BarDetection]:
        """
        Track bar in subsequent frames using optical flow or template matching.

        Args:
            frame: Current frame
            prev_detection: Previous bar detection
            person_mask: Optional person segmentation mask

        Returns:
            Updated BarDetection or None
        """
        # Search in neighborhood of previous detection
        cx, cy = prev_detection.center
        search_radius = 100  # Increased search radius

        x1 = max(0, int(cx - search_radius))
        y1 = max(0, int(cy - search_radius))
        x2 = min(frame.shape[1], int(cx + search_radius))
        y2 = min(frame.shape[0], int(cy + search_radius))

        roi = (x1, y1, x2 - x1, y2 - y1)
        return self.detect_bar_initial(frame, roi, person_mask=person_mask)

    def track_sequence(self, frames: List[np.ndarray], initial_detection: Optional[BarDetection] = None,
                       person_masks: Optional[List[Optional[np.ndarray]]] = None) -> List[Optional[BarDetection]]:
        """
        Track bar across a sequence of frames.

        Args:
            frames: List of frames
            initial_detection: Initial detection or None to auto-detect
            person_masks: Optional list of person segmentation masks

        Returns:
            List of BarDetection objects (one per frame)
        """
        if not frames:
            return []

        detections = []

        # Get mask for first frame
        first_mask = person_masks[0] if person_masks and len(person_masks) > 0 else None

        # Initial detection
        if initial_detection:
            current_detection = initial_detection
        else:
            current_detection = self.detect_bar_initial(frames[0], person_mask=first_mask)

        detections.append(current_detection)

        # Track through remaining frames
        for i, frame in enumerate(frames[1:], 1):
            # Get mask for this frame
            mask = person_masks[i] if person_masks and i < len(person_masks) else None

            if current_detection:
                current_detection = self.track_bar(frame, current_detection, person_mask=mask)
            else:
                # Try to re-detect if tracking lost
                current_detection = self.detect_bar_initial(frame, person_mask=mask)

            detections.append(current_detection)

        return detections

    def compute_bar_trajectory(self, detections: List[Optional[BarDetection]],
                               timestamps: List[float],
                               smooth: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bar position, velocity, and acceleration from detections.

        Args:
            detections: List of BarDetection objects
            timestamps: Corresponding timestamps
            smooth: Whether to smooth the trajectory

        Returns:
            Tuple of (positions, velocities, accelerations) as Nx3 arrays
        """
        # Extract positions
        positions = []
        valid_indices = []

        for i, det in enumerate(detections):
            if det:
                positions.append([det.center[0], det.center[1], 0.0])  # 2D + placeholder z
                valid_indices.append(i)

        if not positions:
            return np.array([]), np.array([]), np.array([])

        positions = np.array(positions)

        # Smooth positions if requested
        if smooth and len(positions) > 5:
            window_length = min(11, len(positions) if len(positions) % 2 == 1 else len(positions) - 1)
            positions[:, 0] = savgol_filter(positions[:, 0], window_length, 3)
            positions[:, 1] = savgol_filter(positions[:, 1], window_length, 3)

        # Compute velocities
        velocities = np.zeros_like(positions)
        if len(positions) > 1:
            for i in range(len(positions)):
                if i == 0:
                    dt = timestamps[valid_indices[1]] - timestamps[valid_indices[0]]
                    velocities[i] = (positions[1] - positions[0]) / (dt + 1e-6)
                elif i == len(positions) - 1:
                    dt = timestamps[valid_indices[i]] - timestamps[valid_indices[i-1]]
                    velocities[i] = (positions[i] - positions[i-1]) / (dt + 1e-6)
                else:
                    dt = timestamps[valid_indices[i+1]] - timestamps[valid_indices[i-1]]
                    velocities[i] = (positions[i+1] - positions[i-1]) / (dt + 1e-6)

        # Compute accelerations
        accelerations = np.zeros_like(positions)
        if len(velocities) > 1:
            for i in range(len(velocities)):
                if i == 0:
                    dt = timestamps[valid_indices[1]] - timestamps[valid_indices[0]]
                    accelerations[i] = (velocities[1] - velocities[0]) / (dt + 1e-6)
                elif i == len(velocities) - 1:
                    dt = timestamps[valid_indices[i]] - timestamps[valid_indices[i-1]]
                    accelerations[i] = (velocities[i] - velocities[i-1]) / (dt + 1e-6)
                else:
                    dt = timestamps[valid_indices[i+1]] - timestamps[valid_indices[i-1]]
                    accelerations[i] = (velocities[i+1] - velocities[i-1]) / (dt + 1e-6)

        return positions, velocities, accelerations

    def draw_detection(self, frame: np.ndarray, detection: BarDetection) -> np.ndarray:
        """
        Draw bar detection on frame.

        Args:
            frame: Input frame
            detection: BarDetection object

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw endpoints
        (x1, y1), (x2, y2) = detection.endpoints
        cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        # Draw center
        cx, cy = detection.center
        cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        # Draw bounding box
        x, y, w, h = detection.bounding_box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Add confidence text
        cv2.putText(annotated, f"Conf: {detection.confidence:.2f}",
                   (int(cx) + 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

        return annotated
