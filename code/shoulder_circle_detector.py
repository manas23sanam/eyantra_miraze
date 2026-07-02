import numpy as np
from typing import Optional, Any, Tuple
from pose_geometry_base import PoseGeometryBase, mp_pose

class EMAFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, new_val: float) -> Optional[float]:
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

class ShoulderCircleDetector(PoseGeometryBase):
    """
    Tracks shoulder rolls (shrugging up, rolling back, down, and forward) from a frontal view.
    Instead of tracking the whole arm, it tracks the shoulder's position relative to the nose 
    (elevation) and the distance between the shoulders (protraction/retraction).
    """
    def __init__(self, depth_intrinsics=None):
        super().__init__(depth_intrinsics=depth_intrinsics)
        self.center_x_ema = EMAFilter(alpha=0.05)
        self.center_y_ema = EMAFilter(alpha=0.05)

    def process_frame(self, color_image: np.ndarray, depth_frame=None, depth_image=None) -> Optional[dict]:
        results = self.run_pose(color_image)
        if not results or not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if nose.visibility < 0.2 or l_shoulder.visibility < 0.2 or r_shoulder.visibility < 0.2:
            return None

        # 1. Chest Width (X axis of our phase space)
        # When rolling shoulders forward, width decreases. Rolling backward, width increases.
        chest_width = abs(l_shoulder.x - r_shoulder.x)

        # 2. Shrug Height (Y axis of our phase space)
        # Y increases downwards in image. 
        # nose.y - mid_shoulder.y is negative. As you shrug up, it gets closer to 0 (increases).
        mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
        shrug_height = nose.y - mid_shoulder_y

        # Update the moving average to find the "center" of the circle
        cx = self.center_x_ema.update(chest_width)
        cy = self.center_y_ema.update(shrug_height)

        if cx is None or cy is None:
            return None

        # Calculate the vector from the center of the roll
        vec_x = chest_width - cx
        vec_y = shrug_height - cy

        # Angle of the shoulder roll phase
        angle_rad = np.arctan2(vec_y, vec_x)
        angle_deg = float(np.degrees(angle_rad)) % 360

        return {
            "angle_deg": angle_deg,
            "chest_width": chest_width,
            "shrug_height": shrug_height,
            "landmarks": results.pose_landmarks
        }


class ShoulderCircleRepCounter:
    """
    A 4-quadrant state machine that requires the shoulder roll phase angle
    to pass through all 4 quadrants sequentially to count a rep.
    """
    def __init__(self):
        self.current_quadrant = None
        self.quadrants_visited = set()
        self.rep_count = 0

    def get_quadrant(self, angle: float) -> int:
        if 0 <= angle < 90: return 1
        elif 90 <= angle < 180: return 2
        elif 180 <= angle < 270: return 3
        else: return 4

    def update(self, angle_deg: Optional[float]) -> Tuple[str, int, bool]:
        just_completed = False
        if angle_deg is None:
            state_str = f"QUADRANT_{self.current_quadrant}" if self.current_quadrant else "NONE"
            return state_str, self.rep_count, just_completed

        new_quadrant = self.get_quadrant(angle_deg)

        if self.current_quadrant is None:
            self.current_quadrant = new_quadrant
            self.quadrants_visited.add(new_quadrant)
        elif new_quadrant != self.current_quadrant:
            # Check for forward progression
            expected_next = (self.current_quadrant % 4) + 1
            if new_quadrant == expected_next:
                self.quadrants_visited.add(new_quadrant)
                self.current_quadrant = new_quadrant

                if len(self.quadrants_visited) == 4:
                    self.rep_count += 1
                    just_completed = True
                    self.quadrants_visited = {new_quadrant}
            else:
                # If they jump randomly or go backwards, reset tracking
                self.quadrants_visited = {new_quadrant}
                self.current_quadrant = new_quadrant

        return f"QUADRANT_{self.current_quadrant}", self.rep_count, just_completed
