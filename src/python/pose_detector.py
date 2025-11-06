"""Pose detection using MediaPipe for tracking lifter kinematics."""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Keypoint:
    """3D keypoint with confidence."""
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth (relative)
    confidence: float


@dataclass
class PoseData:
    """Complete pose data for a single frame."""
    timestamp: float
    keypoints: Dict[str, Keypoint]
    frame_idx: int
    visibility_score: float  # Overall pose visibility
    segmentation_mask: Optional[np.ndarray] = None  # Person segmentation mask


class PoseDetector:
    """Detect and track human pose using MediaPipe."""

    # MediaPipe pose landmark indices
    LANDMARK_NAMES = {
        0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
        4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
        7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        17: 'left_pinky', 18: 'right_pinky',
        19: 'left_index', 20: 'right_index',
        21: 'left_thumb', 22: 'right_thumb',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
        29: 'left_heel', 30: 'right_heel',
        31: 'left_foot_index', 32: 'right_foot_index'
    }

    def __init__(self, model_complexity: int = 2, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose detector.

        Args:
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=True  # For person segmentation
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose(self, frame: np.ndarray, timestamp: float, frame_idx: int) -> Optional[PoseData]:
        """
        Detect pose in a single frame.

        Args:
            frame: Input frame (BGR)
            timestamp: Frame timestamp
            frame_idx: Frame index

        Returns:
            PoseData or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract keypoints
        keypoints = {}
        total_visibility = 0.0

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            name = self.LANDMARK_NAMES.get(idx, f"landmark_{idx}")
            keypoints[name] = Keypoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                confidence=landmark.visibility
            )
            total_visibility += landmark.visibility

        visibility_score = total_visibility / len(results.pose_landmarks.landmark)

        # Extract segmentation mask if available
        segmentation_mask = None
        if results.segmentation_mask is not None:
            segmentation_mask = results.segmentation_mask

        return PoseData(
            timestamp=timestamp,
            keypoints=keypoints,
            frame_idx=frame_idx,
            visibility_score=visibility_score,
            segmentation_mask=segmentation_mask
        )

    def detect_pose_sequence(self, frames: List[Tuple[int, float, np.ndarray]]) -> List[PoseData]:
        """
        Detect poses in a sequence of frames.

        Args:
            frames: List of (frame_idx, timestamp, frame) tuples

        Returns:
            List of PoseData objects
        """
        poses = []

        for frame_idx, timestamp, frame in frames:
            pose_data = self.detect_pose(frame, timestamp, frame_idx)
            if pose_data:
                poses.append(pose_data)

        return poses

    def get_joint_angles(self, pose: PoseData, lift_type: str = 'squat') -> Dict[str, float]:
        """
        Calculate relevant joint angles based on lift type.

        Args:
            pose: PoseData object
            lift_type: Type of lift (squat, bench, deadlift)

        Returns:
            Dictionary of joint angles in radians
        """
        angles = {}

        if lift_type == 'squat':
            # Knee angle
            if all(k in pose.keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
                angles['left_knee'] = self._calculate_angle(
                    pose.keypoints['left_hip'],
                    pose.keypoints['left_knee'],
                    pose.keypoints['left_ankle']
                )

            if all(k in pose.keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
                angles['right_knee'] = self._calculate_angle(
                    pose.keypoints['right_hip'],
                    pose.keypoints['right_knee'],
                    pose.keypoints['right_ankle']
                )

            # Hip angle
            if all(k in pose.keypoints for k in ['left_shoulder', 'left_hip', 'left_knee']):
                angles['left_hip'] = self._calculate_angle(
                    pose.keypoints['left_shoulder'],
                    pose.keypoints['left_hip'],
                    pose.keypoints['left_knee']
                )

        elif lift_type == 'bench':
            # Elbow angle
            if all(k in pose.keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angles['left_elbow'] = self._calculate_angle(
                    pose.keypoints['left_shoulder'],
                    pose.keypoints['left_elbow'],
                    pose.keypoints['left_wrist']
                )

            if all(k in pose.keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                angles['right_elbow'] = self._calculate_angle(
                    pose.keypoints['right_shoulder'],
                    pose.keypoints['right_elbow'],
                    pose.keypoints['right_wrist']
                )

        return angles

    def _calculate_angle(self, p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
        """Calculate angle between three points (in radians)."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])

        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

        # Angle
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return angle

    def draw_pose(self, frame: np.ndarray, pose: PoseData) -> np.ndarray:
        """
        Draw pose landmarks on frame.

        Args:
            frame: Input frame
            pose: PoseData object

        Returns:
            Frame with pose drawn
        """
        annotated = frame.copy()

        # Draw landmarks
        for name, kp in pose.keypoints.items():
            if kp.confidence > 0.5:
                x = int(kp.x * frame.shape[1])
                y = int(kp.y * frame.shape[0])
                cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

        # Draw connections (simplified)
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]

        for start, end in connections:
            if start in pose.keypoints and end in pose.keypoints:
                kp1, kp2 = pose.keypoints[start], pose.keypoints[end]
                if kp1.confidence > 0.5 and kp2.confidence > 0.5:
                    pt1 = (int(kp1.x * frame.shape[1]), int(kp1.y * frame.shape[0]))
                    pt2 = (int(kp2.x * frame.shape[1]), int(kp2.y * frame.shape[0]))
                    cv2.line(annotated, pt1, pt2, (255, 0, 0), 2)

        return annotated

    def get_person_mask(self, pose: PoseData, threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Get binary mask of person from segmentation.

        Args:
            pose: PoseData object with segmentation mask
            threshold: Threshold for binary mask (0-1)

        Returns:
            Binary mask as uint8 (0 or 255) or None if no mask
        """
        if pose.segmentation_mask is None:
            return None

        # Convert to binary mask
        mask = (pose.segmentation_mask > threshold).astype(np.uint8) * 255
        return mask

    def draw_segmentation(self, frame: np.ndarray, pose: PoseData,
                         alpha: float = 0.5, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Overlay segmentation mask on frame.

        Args:
            frame: Input frame
            pose: PoseData object with segmentation mask
            alpha: Transparency of overlay (0-1)
            color: RGB color for person mask

        Returns:
            Frame with segmentation overlay
        """
        if pose.segmentation_mask is None:
            return frame

        annotated = frame.copy()

        # Get binary mask
        mask = self.get_person_mask(pose, threshold=0.5)
        if mask is None:
            return frame

        # Resize mask to frame size if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask > 127] = color

        # Blend with original frame
        annotated = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)

        return annotated

    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose') and self.pose:
            try:
                self.pose.close()
            except (ValueError, AttributeError):
                pass  # Already closed

    def __del__(self):
        """Destructor."""
        try:
            self.release()
        except:
            pass  # Ignore cleanup errors
