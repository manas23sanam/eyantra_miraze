from pose_geometry_base import PoseGeometryBase, mp_pose


class SitToStandDetector(PoseGeometryBase):
    def process_frame(self, color_image, depth_frame, depth_image, side="auto"):
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None

        joint_ids = [
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        joints = self.get_landmarks_3d(results, joint_ids, depth_frame, depth_image)

        left_hip, lh_vis = joints[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee, lk_vis = joints[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle, la_vis = joints[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_hip, rh_vis = joints[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee, rk_vis = joints[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle, ra_vis = joints[mp_pose.PoseLandmark.RIGHT_ANKLE]

        left_angle = (self.angle_between(left_hip, left_knee, left_ankle)
                      if left_hip is not None and left_knee is not None and left_ankle is not None
                      else None)
        right_angle = (self.angle_between(right_hip, right_knee, right_ankle)
                       if right_hip is not None and right_knee is not None and right_ankle is not None
                       else None)

        chosen_angle, chosen_side = None, None
        if side == "left":
            chosen_angle, chosen_side = left_angle, "left"
        elif side == "right":
            chosen_angle, chosen_side = right_angle, "right"
        else:
            left_conf = min(lh_vis, lk_vis, la_vis)
            right_conf = min(rh_vis, rk_vis, ra_vis)
            if left_angle is not None and (right_angle is None or left_conf >= right_conf):
                chosen_angle, chosen_side = left_angle, "left"
            elif right_angle is not None:
                chosen_angle, chosen_side = right_angle, "right"

        if chosen_angle is None:
            return None

        return {
            "hip_knee_ankle_angle_deg": chosen_angle,
            "side_used": chosen_side,
            "landmarks": results.pose_landmarks,
        }


class SitToStandRepCounter:
    """
    Tracks state across frames to count sit-to-stand reps.
    A rep = going from STANDING (angle near 180) down to SITTING
    (angle drops below sit_threshold) and back up to STANDING
    (angle rises back above stand_threshold).

    Use this on top of SitToStandDetector's per-frame angle output.
    """
    def __init__(self, stand_threshold=160.0, sit_threshold=110.0):
        self.stand_threshold = stand_threshold
        self.sit_threshold = sit_threshold
        self.state = "standing"   # "standing" or "sitting"
        self.rep_count = 0

    def update(self, angle):
        """Call once per frame with the current angle. Returns (state, rep_count, just_completed_rep)."""
        just_completed = False
        if angle is None:
            return self.state, self.rep_count, just_completed

        if self.state == "standing" and angle <= self.sit_threshold:
            self.state = "sitting"
        elif self.state == "sitting" and angle >= self.stand_threshold:
            self.state = "standing"
            self.rep_count += 1
            just_completed = True

        return self.state, self.rep_count, just_completed
