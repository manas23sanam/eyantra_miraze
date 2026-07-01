"""
pose_geometry_base.py

Shared base class for ALL exercise detectors (squat, sit-to-stand,
shoulder circle, pendulum, upper trapezius).

Why a shared base: every exercise needs the same two underlying steps -
  1. Run MediaPipe Pose on the color frame to get landmark pixels
  2. Convert any landmark pixel + its depth reading into a real 3D point

Only the THIRD step differs per exercise: which joints to combine and
what angle/pattern to compute from them. That exercise-specific logic
lives in its own file (squat_angle_detector.py, sit_to_stand_detector.py,
etc.) and each one imports and extends this base.

None of this is a trained ML model. With only 4-5 sample videos per new
exercise, training a classifier would just memorize those specific clips
and fail on a live person - not enough data to generalize. Geometry
(joint angles) doesn't need "enough data" the way ML does; it needs
correct anatomy and a sensible threshold, which we validate against
your sample videos instead of training on them.
"""

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


class PoseGeometryBase:
    def __init__(self, depth_intrinsics, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, visibility_floor=0.15):
        """
        depth_intrinsics: rs.intrinsics from your depth stream profile:
            profile = pipeline.start(config)
            depth_stream = profile.get_stream(rs.stream.depth)
            depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        visibility_floor: minimum MediaPipe landmark visibility to accept a
            joint reading. Kept low (0.15) so angled/partially-occluded views
            (e.g. 45-degree shots) still produce a reading from whichever
            joints ARE visible, rather than rejecting the whole frame.
        """
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.intrinsics = depth_intrinsics
        self.visibility_floor = visibility_floor

    def close(self):
        self.pose.close()

    def _pixel_depth_to_3d(self, px, py, depth_frame, depth_image):
        """Pixel (0-1 normalized) + depth -> real-world 3D point in mm."""
        h, w = depth_image.shape
        x = int(np.clip(px * w, 0, w - 1))
        y = int(np.clip(py * h, 0, h - 1))

        depth_mm = depth_frame.get_distance(x, y) * 1000.0
        if depth_mm <= 0:
            return None

        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        X = (x - cx) * depth_mm / fx
        Y = (y - cy) * depth_mm / fy
        Z = depth_mm
        return np.array([X, Y, Z])

    @staticmethod
    def angle_between(a, b, c):
        """Angle at point b formed by a-b-c, in degrees. 180 = straight line."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def get_landmarks_3d(self, results, landmark_ids, depth_frame, depth_image):
        """
        Given MediaPipe results and a list of PoseLandmark ids, returns a dict
        {landmark_id: (point_3d_or_None, visibility)} for each requested joint.
        Centralizes the visibility-floor check so every exercise detector
        applies the same rule consistently.
        """
        lm = results.pose_landmarks.landmark
        out = {}
        for landmark_id in landmark_ids:
            point = lm[landmark_id]
            if point.visibility < self.visibility_floor:
                out[landmark_id] = (None, point.visibility)
            else:
                p3d = self._pixel_depth_to_3d(point.x, point.y, depth_frame, depth_image)
                out[landmark_id] = (p3d, point.visibility)
        return out

    def run_pose(self, color_image):
        """Runs MediaPipe on a BGR color image. Returns MediaPipe results object."""
        rgb_image = color_image[:, :, ::-1]
        return self.pose.process(rgb_image)


    def get_torso_lean_angle(self, landmarks_proto, side="left"):
        """
        Calculates the angle between the shoulder, hip, and a virtual point
        directly above the hip to measure torso lean.
        """
        if landmarks_proto is None:
            return None
            
        landmarks = landmarks_proto.landmark
        
        if side == "left":
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        else:
            hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
        if hip.visibility < 0.2 or shoulder.visibility < 0.2:
            return None
            
        hip_pt = np.array([hip.x, hip.y, hip.z])
        shoulder_pt = np.array([shoulder.x, shoulder.y, shoulder.z])
        vertical_up = np.array([hip.x, hip.y - 0.5, hip.z])
        
        return self.angle_between(shoulder_pt, hip_pt, vertical_up)

class EMAFilter:
    """
    Exponential Moving Average (EMA) filter to smooth out noisy sensor data.
    alpha is the smoothing factor between 0 and 1.
    Higher alpha = less smoothing, more responsive.
    Lower alpha = more smoothing, less responsive.
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return self.value
            
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value
