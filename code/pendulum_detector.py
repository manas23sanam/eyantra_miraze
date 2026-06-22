"""
pendulum_detector.py

Pendulum shoulder exercise: person bends forward at the hips/torso and
lets one arm hang down, then swings it like a pendulum. This is the most
complex of the 5 because it requires checking TWO things at once:

  1. POSTURE CHECK (static, per-frame): is the torso actually bent over
     to roughly the right angle? If someone stands upright and just
     swings their arm, that's not a correct pendulum exercise even if
     the arm is moving - the bent-over position is what allows gravity
     to do the work and is the actual therapeutic point of the exercise.

  2. SWING CHECK (time-series, like shoulder circle): is the arm actually
     swinging back and forth through a reasonable range, rather than
     just hanging still?

Both checks reuse logic you already have:
  - Torso bend angle = same angle_between() math as squats, just applied
    to shoulder-hip-knee instead of hip-knee-ankle.
  - Arm swing = same oscillation-tracking idea as shoulder circle, but
    simpler (back-and-forth, not full rotation) so we track min/max swing
    angle reached rather than accumulated rotation.
"""

import numpy as np
from pose_geometry_base import PoseGeometryBase, mp_pose


class PendulumDetector(PoseGeometryBase):
    def process_frame(self, color_image, depth_frame, depth_image, swinging_side="right"):
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None

        shoulder_id = (mp_pose.PoseLandmark.RIGHT_SHOULDER if swinging_side == "right"
                       else mp_pose.PoseLandmark.LEFT_SHOULDER)
        elbow_id = (mp_pose.PoseLandmark.RIGHT_ELBOW if swinging_side == "right"
                    else mp_pose.PoseLandmark.LEFT_ELBOW)
        hip_id = (mp_pose.PoseLandmark.RIGHT_HIP if swinging_side == "right"
                  else mp_pose.PoseLandmark.LEFT_HIP)
        knee_id = (mp_pose.PoseLandmark.RIGHT_KNEE if swinging_side == "right"
                   else mp_pose.PoseLandmark.LEFT_KNEE)

        joint_ids = [shoulder_id, elbow_id, hip_id, knee_id]
        joints = self.get_landmarks_3d(results, joint_ids, depth_frame, depth_image)

        shoulder, _ = joints[shoulder_id]
        elbow, _ = joints[elbow_id]
        hip, _ = joints[hip_id]
        knee, _ = joints[knee_id]

        if shoulder is None or hip is None or knee is None:
            return None  # need at least these for the posture check

        # ---- 1. POSTURE CHECK: torso bend angle (shoulder-hip-knee) ----
        # Standing straight: ~180 degrees. Bent over at hips: much smaller.
        torso_bend_angle = self.angle_between(shoulder, hip, knee)

        # ---- 2. SWING CHECK: arm angle relative to torso line ----
        arm_swing_angle = None
        if elbow is not None:
            torso_vec = shoulder - hip
            arm_vec = elbow - shoulder
            cos_angle = (np.dot(torso_vec, arm_vec) /
                         (np.linalg.norm(torso_vec) * np.linalg.norm(arm_vec) + 1e-9))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            arm_swing_angle = float(np.degrees(np.arccos(cos_angle)))

        return {
            "torso_bend_angle_deg": torso_bend_angle,
            "arm_swing_angle_deg": arm_swing_angle,
            "landmarks": results.pose_landmarks,
        }


class PendulumSwingTracker:
    """
    Stateful tracker for the arm swing component. Tracks the min/max
    arm_swing_angle seen recently to determine if the person is actually
    swinging through a reasonable range, vs. holding the arm still.
    """
    def __init__(self, window_size=30, min_range_deg=20.0):
        self.window_size = window_size
        self.min_range_deg = min_range_deg
        self.history = []

    def update(self, arm_swing_angle):
        if arm_swing_angle is not None:
            self.history.append(arm_swing_angle)
            if len(self.history) > self.window_size:
                self.history.pop(0)

        if len(self.history) < 5:
            return {"is_swinging": False, "range_deg": 0.0}

        range_deg = max(self.history) - min(self.history)
        return {"is_swinging": range_deg >= self.min_range_deg, "range_deg": range_deg}
