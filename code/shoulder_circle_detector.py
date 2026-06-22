"""
shoulder_circle_detector.py

Shoulder rolls: the arm/shoulder moves in a circular path rather than
holding a static angle, so this is NOT a single-frame angle check like
squats - it's a TIME-SERIES check. We track the angle of the
shoulder-to-elbow vector (relative to a fixed reference) across frames
and look for it sweeping through a full rotation.

Approach: project the shoulder-to-elbow vector onto a 2D plane (using
the torso's coordinate frame so it's somewhat robust to camera angle),
compute its angle each frame, and accumulate angular change over time.
A "correct" circle = the accumulated angle change completes close to
360 degrees in a smooth, consistent direction. A jerky or incomplete
movement (accumulated angle far short of 360, or direction reversals)
gets flagged as incorrect.

This needs a tracker object that persists across frames (not just a
single process_frame call) because "is this a circle" is inherently
a multi-frame question.
"""

import numpy as np
from pose_geometry_base import PoseGeometryBase, mp_pose


class ShoulderCircleDetector(PoseGeometryBase):
    def get_shoulder_elbow_angle(self, color_image, depth_frame, depth_image, side="left"):
        """
        Returns the angle (in degrees, 0-360) of the shoulder->elbow vector
        within the frontal plane (X-Y, ignoring depth Z) for one frame.
        Returns None if landmarks aren't visible.
        """
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None, None

        shoulder_id = (mp_pose.PoseLandmark.LEFT_SHOULDER if side == "left"
                        else mp_pose.PoseLandmark.RIGHT_SHOULDER)
        elbow_id = (mp_pose.PoseLandmark.LEFT_ELBOW if side == "left"
                    else mp_pose.PoseLandmark.RIGHT_ELBOW)

        joints = self.get_landmarks_3d(results, [shoulder_id, elbow_id], depth_frame, depth_image)
        shoulder, _ = joints[shoulder_id]
        elbow, _ = joints[elbow_id]

        if shoulder is None or elbow is None:
            return None, results.pose_landmarks

        vec = elbow - shoulder
        angle_rad = np.arctan2(vec[1], vec[0])  # frontal plane angle
        angle_deg = float(np.degrees(angle_rad)) % 360

        return angle_deg, results.pose_landmarks


class ShoulderCircleTracker:
    """
    Stateful tracker - call update() once per frame with the current
    shoulder-elbow angle. Accumulates total rotation and detects
    direction reversals, so you can tell whether a full smooth circle
    was completed.
    """
    def __init__(self, completion_threshold_deg=300.0, reversal_tolerance_deg=15.0):
        self.completion_threshold_deg = completion_threshold_deg
        self.reversal_tolerance_deg = reversal_tolerance_deg
        self.prev_angle = None
        self.accumulated_rotation = 0.0
        self.direction = None  # "cw" or "ccw"
        self.reversal_count = 0

    def update(self, angle_deg):
        """
        Returns dict: {accumulated_rotation, is_complete, reversal_count, direction}
        """
        if angle_deg is None:
            return self._status()

        if self.prev_angle is None:
            self.prev_angle = angle_deg
            return self._status()

        diff = angle_deg - self.prev_angle
        # Handle wraparound (e.g. 359 -> 2 degrees should be +3, not -357)
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360

        current_direction = "ccw" if diff > 0 else "cw"
        if self.direction is None:
            self.direction = current_direction
        elif current_direction != self.direction and abs(diff) > self.reversal_tolerance_deg:
            self.reversal_count += 1
            self.direction = current_direction

        self.accumulated_rotation += abs(diff)
        self.prev_angle = angle_deg

        return self._status()

    def _status(self):
        return {
            "accumulated_rotation_deg": self.accumulated_rotation,
            "is_complete": self.accumulated_rotation >= self.completion_threshold_deg,
            "reversal_count": self.reversal_count,
            "direction": self.direction,
        }

    def reset(self):
        self.prev_angle = None
        self.accumulated_rotation = 0.0
        self.direction = None
        self.reversal_count = 0
