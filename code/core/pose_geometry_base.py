import numpy as np
import mediapipe as mp
from typing import Optional, Any
mp_pose = mp.solutions.pose

class PoseGeometryBase:

    def __init__(self, depth_intrinsics: Any=None, min_detection_confidence: float=0.5, min_tracking_confidence: float=0.5, visibility_floor: float=0.15):
        self.pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.intrinsics = depth_intrinsics
        self.visibility_floor = visibility_floor

    def close(self):
        self.pose.close()

    def _pixel_depth_to_3d(self, px: float, py: float, depth_frame: Any, depth_image: np.ndarray) -> Optional[np.ndarray]:
        h, w = depth_image.shape
        x = int(np.clip(px * w, 0, w - 1))
        y = int(np.clip(py * h, 0, h - 1))
        depth_mm = depth_frame.get_distance(x, y) * 1000.0
        if depth_mm <= 0:
            return None
        fx, fy = (self.intrinsics.fx, self.intrinsics.fy)
        cx, cy = (self.intrinsics.ppx, self.intrinsics.ppy)
        X = (x - cx) * depth_mm / fx
        Y = (y - cy) * depth_mm / fy
        Z = depth_mm
        return np.array([X, Y, Z])

    @staticmethod
    def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-09)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def get_landmarks_3d(self, results: Any, landmark_ids: list[int], depth_frame: Any, depth_image: np.ndarray) -> dict[int, tuple[Optional[np.ndarray], float]]:
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

    def run_pose(self, color_image: np.ndarray) -> Any:
        rgb_image = color_image[:, :, ::-1]
        return self.pose.process(rgb_image)

    def get_torso_lean_angle(self, landmarks_proto: Any, side: str='left') -> Optional[float]:
        if landmarks_proto is None:
            return None
        landmarks = landmarks_proto.landmark
        if side == 'left':
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

    def __init__(self, alpha: float=0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: Optional[float]) -> Optional[float]:
        if new_value is None:
            return self.value
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value
