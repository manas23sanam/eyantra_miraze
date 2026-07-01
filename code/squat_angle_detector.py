import numpy as np
from typing import Optional, Any, Tuple
from pose_geometry_base import PoseGeometryBase, mp_pose

class SquatAngleDetector(PoseGeometryBase):
    def process_frame(self, color_image: np.ndarray, depth_frame: Any, depth_image: np.ndarray, side: str = "auto", visibility_floor: float = 0.15) -> Optional[dict]:
        """
        Run pose detection on one color frame and compute the knee angle(s).

        side: "left", "right", or "auto" (uses whichever leg has higher
              MediaPipe visibility confidence - useful for side-view shots
              where one leg is closer to the camera than the other).

        visibility_floor: minimum MediaPipe landmark visibility to accept a
              joint.

        Returns a dict with angle info, or None if pose/landmarks weren't found.
        """
        # Set visibility floor for this frame
        self.visibility_floor = visibility_floor
        
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None

        joint_ids = [
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        joints = self.get_landmarks_3d(results, joint_ids, depth_frame, depth_image)

        left_hip, lh_vis = joints[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee, lk_vis = joints[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle, la_vis = joints[mp_pose.PoseLandmark.LEFT_ANKLE]

        right_hip, rh_vis = joints[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee, rk_vis = joints[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle, ra_vis = joints[mp_pose.PoseLandmark.RIGHT_ANKLE]

        left_angle = None
        right_angle = None

        if left_hip is not None and left_knee is not None and left_ankle is not None:
            left_angle = self.angle_between(left_hip, left_knee, left_ankle)

        if right_hip is not None and right_knee is not None and right_ankle is not None:
            right_angle = self.angle_between(right_hip, right_knee, right_ankle)

        # Calculate knee distance
        knee_dist_mm = None
        if left_knee is not None and right_knee is not None:
            knee_dist_mm = float(np.linalg.norm(left_knee - right_knee))

        # Calculate angle difference
        angle_diff = None
        if left_angle is not None and right_angle is not None:
            angle_diff = abs(left_angle - right_angle)

        # Calculate vertical ratio (femur Y drop / shin Y drop)
        left_ratio = None
        if left_hip is not None and left_knee is not None and left_ankle is not None:
            femur_y = abs(left_hip[1] - left_knee[1])
            shin_y = abs(left_knee[1] - left_ankle[1])
            left_ratio = femur_y / (shin_y + 1e-6)

        right_ratio = None
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            femur_y = abs(right_hip[1] - right_knee[1])
            shin_y = abs(right_knee[1] - right_ankle[1])
            right_ratio = femur_y / (shin_y + 1e-6)

        chosen_angle = None
        chosen_side = None
        chosen_ratio = None

        if side == "left":
            chosen_angle, chosen_side, chosen_ratio = left_angle, "left", left_ratio
        elif side == "right":
            chosen_angle, chosen_side, chosen_ratio = right_angle, "right", right_ratio
        else:  # auto: prefer the side with higher landmark visibility
            left_conf = min(lh_vis, lk_vis, la_vis)
            right_conf = min(rh_vis, rk_vis, ra_vis)
            if left_angle is not None and (right_angle is None or left_conf >= right_conf):
                chosen_angle, chosen_side, chosen_ratio = left_angle, "left", left_ratio
            elif right_angle is not None:
                chosen_angle, chosen_side, chosen_ratio = right_angle, "right", right_ratio

        if chosen_angle is None:
            return None

        # Calculate torso lean for the chosen side
        torso_lean_deg = self.get_torso_lean_angle(results.pose_landmarks, side=chosen_side)

        return {
            "knee_angle_deg": chosen_angle,
            "vertical_ratio": chosen_ratio,
            "torso_lean_deg": torso_lean_deg,
            "side_used": chosen_side,
            "left_angle_deg": left_angle,
            "right_angle_deg": right_angle,
            "knee_dist_mm": knee_dist_mm,
            "angle_diff": angle_diff,
            "landmarks": results.pose_landmarks,  # for drawing/visualization later
        }

class SquatRepCounter:
    """
    Tracks state across frames to count squat reps.
    A rep = going from STANDING down to SQUATTING (crossing squat_threshold)
    and back up to STANDING (crossing stand_threshold).
    If the user drops below angle_min, spreads knees too wide, squats unevenly,
    or leans their back too far, the rep is marked as INVALID and will not count.
    """
    def __init__(self, stand_threshold: float = 160.0, squat_threshold: float = 135.0, angle_min: Optional[float] = None,
                 max_angle_diff: Optional[float] = None, max_knee_dist_mm: Optional[float] = None, max_torso_lean: Optional[float] = None):
        self.stand_threshold = stand_threshold
        self.squat_threshold = squat_threshold
        self.angle_min = angle_min
        self.max_angle_diff = max_angle_diff
        self.max_knee_dist_mm = max_knee_dist_mm
        self.max_torso_lean = max_torso_lean
        
        self.state = "standing"
        self.rep_count = 0

    def update(self, metric: Optional[float], angle_diff: Optional[float] = None, knee_dist: Optional[float] = None, torso_lean: Optional[float] = None) -> Tuple[str, int, bool]:
        """Call once per frame with current metrics. Returns (state, rep_count, just_completed_rep)."""
        just_completed = False
        if metric is None:
            return self.state, self.rep_count, just_completed

        if self.state == "standing" and metric <= self.squat_threshold:
            self.state = "squatting"
            
        elif self.state == "squatting":
            # Check form constraints
            if self.angle_min is not None and metric < self.angle_min:
                self.state = "invalid_depth"
            elif self.max_angle_diff is not None and angle_diff is not None and angle_diff > self.max_angle_diff:
                self.state = "invalid_asymmetric"
            elif self.max_knee_dist_mm is not None and knee_dist is not None and knee_dist > self.max_knee_dist_mm:
                self.state = "invalid_wide_knees"
            elif self.max_torso_lean is not None and torso_lean is not None and torso_lean > self.max_torso_lean:
                self.state = "invalid_back_posture"
            # If form is good and we stand up
            elif metric >= self.stand_threshold:
                self.state = "standing"
                self.rep_count += 1
                just_completed = True
                
        elif self.state in ["invalid_depth", "invalid_asymmetric", "invalid_wide_knees", "invalid_back_posture"]:
            if metric >= self.stand_threshold:
                self.state = "standing"  # Reset back to standing, but DO NOT count the rep

        return self.state, self.rep_count, just_completed