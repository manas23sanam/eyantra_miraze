import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


class SquatAngleDetector:
    def __init__(self, depth_intrinsics, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        depth_intrinsics: rs.intrinsics object from your depth stream profile.
            Get this with:
                profile = pipeline.start(config)
                depth_stream = profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        """
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.intrinsics = depth_intrinsics

    def close(self):
        self.pose.close()

    def _pixel_depth_to_3d(self, px, py, depth_frame, depth_image):
        """
        Convert a 2D pixel (px, py) + its depth value into a real-world
        3D point in millimeters, using the camera's intrinsics.
        Returns None if depth at that pixel is invalid (0 / out of range).
        """
        h, w = depth_image.shape
        x = int(np.clip(px * w, 0, w - 1))
        y = int(np.clip(py * h, 0, h - 1))

        depth_mm = depth_frame.get_distance(x, y) * 1000.0  # meters -> mm
        if depth_mm <= 0:
            return None

        # Standard pinhole back-projection using intrinsics
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        X = (x - cx) * depth_mm / fx
        Y = (y - cy) * depth_mm / fy
        Z = depth_mm
        return np.array([X, Y, Z])

    @staticmethod
    def _angle_between(a, b, c):
        """
        Angle at point b, formed by points a-b-c, in degrees.
        a, b, c are 3D numpy arrays (X, Y, Z in mm).
        180 degrees = fully straight leg.
        Smaller angle = deeper bend.
        """
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def process_frame(self, color_image, depth_frame, depth_image, side="auto",
                       visibility_floor=0.15):
        """
        Run pose detection on one color frame and compute the knee angle(s).

        side: "left", "right", or "auto" (uses whichever leg has higher
              MediaPipe visibility confidence - useful for side-view shots
              where one leg is closer to the camera than the other).

        visibility_floor: minimum MediaPipe landmark visibility to accept a
              joint. Lowered from a stricter default so 45-degree / angled
              views (where one leg partially occludes the other) still
              produce a reading from whichever leg IS visible, instead of
              silently returning None for the whole frame.

        Returns a dict with angle info, or None if pose/landmarks weren't found.
        """
        rgb_image = color_image[:, :, ::-1]  # BGR -> RGB for MediaPipe
        results = self.pose.process(rgb_image)

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        def get_landmark_3d(landmark_id):
            point = lm[landmark_id]
            if point.visibility < visibility_floor:
                return None, point.visibility
            p3d = self._pixel_depth_to_3d(point.x, point.y, depth_frame, depth_image)
            return p3d, point.visibility

        left_hip, lh_vis = get_landmark_3d(mp_pose.PoseLandmark.LEFT_HIP)
        left_knee, lk_vis = get_landmark_3d(mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle, la_vis = get_landmark_3d(mp_pose.PoseLandmark.LEFT_ANKLE)

        right_hip, rh_vis = get_landmark_3d(mp_pose.PoseLandmark.RIGHT_HIP)
        right_knee, rk_vis = get_landmark_3d(mp_pose.PoseLandmark.RIGHT_KNEE)
        right_ankle, ra_vis = get_landmark_3d(mp_pose.PoseLandmark.RIGHT_ANKLE)

        left_angle = None
        right_angle = None

        if left_hip is not None and left_knee is not None and left_ankle is not None:
            left_angle = self._angle_between(left_hip, left_knee, left_ankle)

        if right_hip is not None and right_knee is not None and right_ankle is not None:
            right_angle = self._angle_between(right_hip, right_knee, right_ankle)

        chosen_angle = None
        chosen_side = None

        if side == "left":
            chosen_angle, chosen_side = left_angle, "left"
        elif side == "right":
            chosen_angle, chosen_side = right_angle, "right"
        else:  # auto: prefer the side with higher landmark visibility
            left_conf = min(lh_vis, lk_vis, la_vis)
            right_conf = min(rh_vis, rk_vis, ra_vis)
            if left_angle is not None and (right_angle is None or left_conf >= right_conf):
                chosen_angle, chosen_side = left_angle, "left"
            elif right_angle is not None:
                chosen_angle, chosen_side = right_angle, "right"

        if chosen_angle is None:
            return None

        return {
            "knee_angle_deg": chosen_angle,
            "side_used": chosen_side,
            "left_angle_deg": left_angle,
            "right_angle_deg": right_angle,
            "landmarks": results.pose_landmarks,  # for drawing/visualization later
        }