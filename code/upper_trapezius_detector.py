"""
upper_trapezius_detector.py

Upper trap stretch: bending the head toward the left or right shoulder.
This is measured differently from the leg exercises - instead of a joint
bend angle, we measure the TILT of the head relative to the shoulder
line. When upright, the line from ear to ear (or nose to mid-shoulder)
is roughly vertical/perpendicular to the shoulder line. Tilting the head
sideways changes that angle.

Method: use the vector from mid-shoulder-point to nose, and compare it
to "straight up" (the vector from mid-shoulder to mid-hip, which gives
you the person's actual torso vertical, accounting for their stance/
camera angle rather than assuming a perfectly upright frame).

This needs the shoulder line and a head landmark, not legs - so it works
fine even if the lower body is out of frame, which is realistic since
this stretch is usually filmed from the chest up.
"""

import numpy as np
from pose_geometry_base import PoseGeometryBase, mp_pose


class UpperTrapeziusDetector(PoseGeometryBase):
    def process_frame(self, color_image, depth_frame, depth_image):
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None

        joint_ids = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        joints = self.get_landmarks_3d(results, joint_ids, depth_frame, depth_image)

        nose, nose_vis = joints[mp_pose.PoseLandmark.NOSE]
        l_shoulder, ls_vis = joints[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder, rs_vis = joints[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip, lh_vis = joints[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip, rh_vis = joints[mp_pose.PoseLandmark.RIGHT_HIP]

        if nose is None or l_shoulder is None or r_shoulder is None:
            return None

        mid_shoulder = (l_shoulder + r_shoulder) / 2.0

        # Reference "straight up" direction: prefer using mid-hip if visible
        # (accounts for camera tilt / person's actual stance), otherwise
        # fall back to a world-vertical assumption.
        if l_hip is not None and r_hip is not None:
            mid_hip = (l_hip + r_hip) / 2.0
            torso_up_ref = mid_shoulder - mid_hip
        else:
            torso_up_ref = np.array([0.0, -1.0, 0.0])  # fallback: image "up"

        head_vector = nose - mid_shoulder

        # Angle between head_vector and torso_up_ref tells us how far the
        # head has tilted from neutral. We also need LEFT/RIGHT direction,
        # which we get from the sign of the horizontal (X) component of the
        # head vector relative to the shoulder line.
        cos_angle = (np.dot(head_vector, torso_up_ref) /
                     (np.linalg.norm(head_vector) * np.linalg.norm(torso_up_ref) + 1e-9))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        tilt_from_vertical_deg = float(np.degrees(np.arccos(cos_angle)))

        shoulder_vector = r_shoulder - l_shoulder
        # Project head_vector onto the shoulder line direction to determine
        # left/right lean sign (positive = toward right shoulder, etc.)
        # This is a simplification - good enough for left/right cue, not
        # claiming clinical precision.
        lean_sign = np.dot(head_vector, shoulder_vector)
        direction = "right" if lean_sign > 0 else "left"

        return {
            "neck_tilt_deg": tilt_from_vertical_deg,
            "tilt_direction": direction,
            "landmarks": results.pose_landmarks,
        }
