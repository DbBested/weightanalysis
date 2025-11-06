#!/usr/bin/env python3
"""Test script to verify MediaPipe segmentation is working."""

import sys
import cv2
sys.path.insert(0, 'src/python')

from pose_detector import PoseDetector
from video_processor import VideoProcessor

def test_segmentation(video_path: str, output_path: str = 'segmentation_test.jpg'):
    """Test segmentation on first frame of video."""

    print("Loading video...")
    with VideoProcessor(video_path) as video:
        # Get first frame
        frame = video.get_frame_at_index(0)
        if frame is None:
            print("Failed to read frame")
            return False

    print("Running pose detection with segmentation...")
    detector = PoseDetector(model_complexity=2)
    pose_data = detector.detect_pose(frame, 0.0, 0)

    if pose_data is None:
        print("No pose detected")
        return False

    print(f"Pose detected with {len(pose_data.keypoints)} keypoints")
    print(f"Visibility score: {pose_data.visibility_score:.2f}")

    # Check segmentation
    if pose_data.segmentation_mask is None:
        print("WARNING: Segmentation mask is None!")
        print("Segmentation may not be working properly.")
        return False

    print(f"Segmentation mask shape: {pose_data.segmentation_mask.shape}")
    print(f"Segmentation mask range: [{pose_data.segmentation_mask.min():.3f}, {pose_data.segmentation_mask.max():.3f}]")

    # Get binary person mask
    person_mask = detector.get_person_mask(pose_data, threshold=0.5)
    if person_mask is not None:
        person_pixels = (person_mask > 0).sum()
        total_pixels = person_mask.size
        print(f"Person occupies {person_pixels/total_pixels*100:.1f}% of frame")

    # Create visualization
    print("Creating visualization...")

    # Draw pose
    viz_pose = detector.draw_pose(frame, pose_data)

    # Draw segmentation overlay
    viz_seg = detector.draw_segmentation(frame, pose_data, alpha=0.3, color=(0, 255, 0))

    # Combine side by side
    viz = cv2.hconcat([viz_pose, viz_seg])

    # Add labels
    cv2.putText(viz, "Pose Keypoints", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(viz, "Segmentation Mask", (frame.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save
    cv2.imwrite(output_path, viz)
    print(f"\nSaved visualization to: {output_path}")
    print(f"Open it to see pose keypoints (left) and segmentation mask (right)")

    # Also save just the mask for inspection
    if person_mask is not None:
        mask_path = output_path.replace('.jpg', '_mask.jpg')
        cv2.imwrite(mask_path, person_mask)
        print(f"Saved binary mask to: {mask_path}")

    detector.release()

    print("\nSUCCESS: Segmentation is working!")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 test_segmentation.py VIDEO_PATH")
        print("\nExample:")
        print("  python3 test_segmentation.py /Users/thomasli/Downloads/225bench.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    success = test_segmentation(video_path)
    sys.exit(0 if success else 1)
