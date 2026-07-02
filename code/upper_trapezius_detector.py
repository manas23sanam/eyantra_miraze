import numpy as np
from typing import Optional, Any, Tuple
from pose_geometry_base import PoseGeometryBase, mp_pose

class UpperTrapeziusDetector(PoseGeometryBase):
    def process_frame(self, color_image: np.ndarray, depth_frame: Any, depth_image: np.ndarray) -> Optional[dict]:
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None

        nose = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x, 
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y])
        l_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        r_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        
        # Check visibility
        if (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility < 0.2 or
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility < 0.2 or
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility < 0.2):
            return None

        # Correct for aspect ratio to get true image-plane angles
        aspect_ratio = 640.0 / 480.0
        nose[0] *= aspect_ratio
        l_shoulder[0] *= aspect_ratio
        r_shoulder[0] *= aspect_ratio

        mid_shoulder = (l_shoulder + r_shoulder) / 2.0

        # In 2D, "up" is negative Y. 
        torso_up_ref = np.array([0.0, -1.0]) 

        head_vector = nose - mid_shoulder

        cos_angle = (np.dot(head_vector, torso_up_ref) /
                     (np.linalg.norm(head_vector) * np.linalg.norm(torso_up_ref) + 1e-9))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        tilt_from_vertical_deg = float(np.degrees(np.arccos(cos_angle)))

        shoulder_vector = r_shoulder - l_shoulder
        lean_sign = np.dot(head_vector, shoulder_vector)
        direction = "right" if lean_sign > 0 else "left"

        return {
            "neck_tilt_deg": tilt_from_vertical_deg,
            "tilt_direction": direction,
            "landmarks": results.pose_landmarks,
        }

class UpperTrapRepCounter:
    """
    State machine for Upper Trapezius stretch.
    States: 'upright', 'stretching_left', 'stretching_right'
    A rep is counted when the neck tilts past stretch_threshold in either direction,
    and then returns to <= upright_threshold.
    """
    def __init__(self, stretch_threshold: float = 30.0, upright_threshold: float = 15.0):
        self.stretch_threshold = stretch_threshold
        self.upright_threshold = upright_threshold
        
        self.state = "upright"
        self.rep_count = 0

    def update(self, neck_tilt: Optional[float], direction: Optional[str]) -> Tuple[str, int, bool]:
        just_completed = False
        
        if neck_tilt is None or direction is None:
            return self.state, self.rep_count, just_completed

        if self.state == "upright":
            if neck_tilt >= self.stretch_threshold:
                self.state = f"stretching_{direction}"
                
        elif self.state.startswith("stretching_"):
            if neck_tilt <= self.upright_threshold:
                self.state = "upright"
                self.rep_count += 1
                just_completed = True

        return self.state, self.rep_count, just_completed
