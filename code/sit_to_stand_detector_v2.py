from typing import Optional, Any, Tuple
import numpy as np
from pose_geometry_base import PoseGeometryBase, mp_pose

class SitToStandDetectorV2(PoseGeometryBase):
    def process_frame(self, color_image: np.ndarray, depth_frame: Any, depth_image: np.ndarray, side: str = "auto") -> Optional[dict]:
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None

        joint_ids = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST,
        ]
        joints = self.get_landmarks_3d(results, joint_ids, depth_frame, depth_image)

        ls, ls_vis = joints[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lh, lh_vis = joints[mp_pose.PoseLandmark.LEFT_HIP]
        lk, lk_vis = joints[mp_pose.PoseLandmark.LEFT_KNEE]
        la, la_vis = joints[mp_pose.PoseLandmark.LEFT_ANKLE]
        le, le_vis = joints[mp_pose.PoseLandmark.LEFT_ELBOW]

        rs, rs_vis = joints[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        rh, rh_vis = joints[mp_pose.PoseLandmark.RIGHT_HIP]
        rk, rk_vis = joints[mp_pose.PoseLandmark.RIGHT_KNEE]
        ra, ra_vis = joints[mp_pose.PoseLandmark.RIGHT_ANKLE]
        re, re_vis = joints[mp_pose.PoseLandmark.RIGHT_ELBOW]

        def calc_angles(s, h, k, a, e):
            leg_angle = self.angle_between(h, k, a) if (h is not None and k is not None and a is not None) else None
            back_angle = self.angle_between(s, h, k) if (s is not None and h is not None and k is not None) else None
            arm_angle = self.angle_between(e, s, h) if (e is not None and s is not None and h is not None) else None
            return leg_angle, back_angle, arm_angle

        l_leg, l_back, l_arm = calc_angles(ls, lh, lk, la, le)
        r_leg, r_back, r_arm = calc_angles(rs, rh, rk, ra, re)

        chosen_leg, chosen_back, chosen_arm, chosen_side = None, None, None, None

        if side == "left":
            chosen_leg, chosen_back, chosen_arm, chosen_side = l_leg, l_back, l_arm, "left"
        elif side == "right":
            chosen_leg, chosen_back, chosen_arm, chosen_side = r_leg, r_back, r_arm, "right"
        else:
            left_conf = min(lh_vis, lk_vis, la_vis)
            right_conf = min(rh_vis, rk_vis, ra_vis)
            if l_leg is not None and (r_leg is None or left_conf >= right_conf):
                chosen_leg, chosen_back, chosen_arm, chosen_side = l_leg, l_back, l_arm, "left"
            elif r_leg is not None:
                chosen_leg, chosen_back, chosen_arm, chosen_side = r_leg, r_back, r_arm, "right"

        if chosen_leg is None:
            return None

        return {
            "leg_angle_deg": chosen_leg,
            "back_angle_deg": chosen_back,
            "arm_angle_deg": chosen_arm,
            "side_used": chosen_side,
            "landmarks": results.pose_landmarks,
        }

class SitToStandRepCounter:
    """
    State machine for Sit-to-Stand.
    States: 'sitting', 'standing'
    A rep is counted when transitioning from sitting to standing and back to sitting.
    """
    def __init__(self, leg_stand_threshold=160.0, leg_sit_threshold=110.0,
                 back_min_angle=120.0, arm_min_angle=70.0, arm_max_angle=110.0):
        self.leg_stand_threshold = leg_stand_threshold
        self.leg_sit_threshold = leg_sit_threshold
        self.back_min_angle = back_min_angle
        self.arm_min_angle = arm_min_angle
        self.arm_max_angle = arm_max_angle
        
        self.state = "sitting"
        self.rep_count = 0

    def update(self, leg_angle: Optional[float], back_angle: Optional[float] = None, arm_angle: Optional[float] = None) -> Tuple[str, int, bool]:
        just_completed = False
        
        if leg_angle is None:
            return self.state, self.rep_count, False

        if self.state == "sitting":
            if leg_angle >= self.leg_stand_threshold:
                self.state = "standing"
        
        elif self.state == "standing":
            if leg_angle <= self.leg_sit_threshold:
                self.state = "sitting"
                self.rep_count += 1
                just_completed = True

        return self.state, self.rep_count, just_completed
