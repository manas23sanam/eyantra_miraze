from core.base_exercise import BaseExercise
from core.pose_geometry_base import EMAFilter
import numpy as np
from typing import Optional, Any, Tuple
from core.pose_geometry_base import PoseGeometryBase, mp_pose

class UpperTrapeziusDetector(PoseGeometryBase):

    def process_frame(self, color_image: np.ndarray, depth_frame: Any, depth_image: np.ndarray) -> Optional[dict]:
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None
        nose = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y])
        l_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        r_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility < 0.2 or results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility < 0.2 or results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility < 0.2:
            return None
        aspect_ratio = 640.0 / 480.0
        nose[0] *= aspect_ratio
        l_shoulder[0] *= aspect_ratio
        r_shoulder[0] *= aspect_ratio
        mid_shoulder = (l_shoulder + r_shoulder) / 2.0
        torso_up_ref = np.array([0.0, -1.0])
        head_vector = nose - mid_shoulder
        cos_angle = np.dot(head_vector, torso_up_ref) / (np.linalg.norm(head_vector) * np.linalg.norm(torso_up_ref) + 1e-09)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        tilt_from_vertical_deg = float(np.degrees(np.arccos(cos_angle)))
        shoulder_vector = r_shoulder - l_shoulder
        lean_sign = np.dot(head_vector, shoulder_vector)
        direction = 'right' if lean_sign > 0 else 'left'
        return {'neck_tilt_deg': tilt_from_vertical_deg, 'tilt_direction': direction, 'landmarks': results.pose_landmarks}

class UpperTrapRepCounter:

    def __init__(self, stretch_threshold: float=30.0, upright_threshold: float=15.0):
        self.stretch_threshold = stretch_threshold
        self.upright_threshold = upright_threshold
        self.state = 'upright'
        self.rep_count = 0

    def update(self, neck_tilt: Optional[float], direction: Optional[str]) -> Tuple[str, int, bool]:
        just_completed = False
        if neck_tilt is None or direction is None:
            return (self.state, self.rep_count, just_completed)
        if self.state == 'upright':
            if neck_tilt >= self.stretch_threshold:
                self.state = f'stretching_{direction}'
        elif self.state.startswith('stretching_'):
            if neck_tilt <= self.upright_threshold:
                self.state = 'upright'
                self.rep_count += 1
                just_completed = True
        return (self.state, self.rep_count, just_completed)

class UpperTrapeziusExercise(BaseExercise):
    def __init__(self, config, intrinsics):
        super().__init__(config, intrinsics)
        self.detector = UpperTrapeziusDetector(depth_intrinsics=intrinsics)
        self.counter = UpperTrapRepCounter()
        self.angle_smoother = EMAFilter(alpha=0.3)

    def process_frame(self, color_image, depth_frame, depth_image):
        result = self.detector.process_frame(color_image, depth_frame, depth_image)
        is_correct = False
        cue_text = 'Step into frame'
        state_text = 'None'
        just_completed = False
        rep_count = self.counter.rep_count
        smoothed_metric = None
        metric_name = 'Neck Tilt'
        landmarks = None
        
        if not result:
            return {
                'metric_value': smoothed_metric,
                'metric_name': metric_name,
                'is_correct': is_correct,
                'cue_text': cue_text,
                'state_text': state_text,
                'rep_count': rep_count,
                'just_completed': just_completed,
                'landmarks': landmarks
            }
            
        landmarks = result.get('landmarks')
        raw_metric = result.get('neck_tilt_deg')
        smoothed_metric = self.angle_smoother.update(raw_metric)
        
        if smoothed_metric is not None:
            state_text, rep_count, just_completed = self.counter.update(smoothed_metric, result.get('tilt_direction'))
            is_correct = True
            if state_text == 'upright':
                cue_text = 'STRETCH NECK'
            else:
                cue_text = 'GOOD STRETCH'
                
        return {
            'metric_value': smoothed_metric,
            'metric_name': metric_name,
            'is_correct': is_correct,
            'cue_text': cue_text,
            'state_text': state_text,
            'rep_count': rep_count,
            'just_completed': just_completed,
            'landmarks': landmarks
        }
