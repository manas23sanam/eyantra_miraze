from core.base_exercise import BaseExercise
from core.pose_geometry_base import EMAFilter
import numpy as np
from typing import Optional, Any, Tuple
from core.pose_geometry_base import PoseGeometryBase, mp_pose

class PendulumDetector(PoseGeometryBase):

    def process_frame(self, color_image: np.ndarray, depth_frame: Any, depth_image: np.ndarray, swinging_side: str='right') -> Optional[dict]:
        results = self.run_pose(color_image)
        if not results.pose_landmarks:
            return None
        shoulder_id = mp_pose.PoseLandmark.RIGHT_SHOULDER if swinging_side == 'right' else mp_pose.PoseLandmark.LEFT_SHOULDER
        elbow_id = mp_pose.PoseLandmark.RIGHT_ELBOW if swinging_side == 'right' else mp_pose.PoseLandmark.LEFT_ELBOW
        hip_id = mp_pose.PoseLandmark.RIGHT_HIP if swinging_side == 'right' else mp_pose.PoseLandmark.LEFT_HIP
        knee_id = mp_pose.PoseLandmark.RIGHT_KNEE if swinging_side == 'right' else mp_pose.PoseLandmark.LEFT_KNEE
        joint_ids = [shoulder_id, elbow_id, hip_id, knee_id]
        joints = self.get_landmarks_3d(results, joint_ids, depth_frame, depth_image)
        shoulder, _ = joints[shoulder_id]
        elbow, _ = joints[elbow_id]
        hip, _ = joints[hip_id]
        knee, _ = joints[knee_id]
        if shoulder is None or hip is None or knee is None:
            return None
        torso_bend_angle = self.angle_between(shoulder, hip, knee)
        arm_swing_angle = None
        shoulder_lm = results.pose_landmarks.landmark[shoulder_id]
        hip_lm = results.pose_landmarks.landmark[hip_id]
        elbow_lm = results.pose_landmarks.landmark[elbow_id]
        if shoulder_lm.visibility > 0.2 and hip_lm.visibility > 0.2 and (elbow_lm.visibility > 0.2):
            aspect_ratio = 640.0 / 480.0
            s2d = np.array([shoulder_lm.x * aspect_ratio, shoulder_lm.y])
            h2d = np.array([hip_lm.x * aspect_ratio, hip_lm.y])
            e2d = np.array([elbow_lm.x * aspect_ratio, elbow_lm.y])
            torso_vec = s2d - h2d
            arm_vec = e2d - s2d
            cos_angle = np.dot(torso_vec, arm_vec) / (np.linalg.norm(torso_vec) * np.linalg.norm(arm_vec) + 1e-09)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            arm_swing_angle = float(np.degrees(np.arccos(cos_angle)))
        return {'torso_bend_angle_deg': torso_bend_angle, 'arm_swing_angle_deg': arm_swing_angle, 'landmarks': results.pose_landmarks}

class PendulumRepCounter:

    def __init__(self, forward_threshold: float=65.0, backward_threshold: float=115.0, neutral_min: float=75.0, neutral_max: float=105.0, max_torso_angle: float=150.0):
        self.forward_threshold = forward_threshold
        self.backward_threshold = backward_threshold
        self.neutral_min = neutral_min
        self.neutral_max = neutral_max
        self.max_torso_angle = max_torso_angle
        self.state = 'neutral'
        self.rep_count = 0
        self.has_swung_forward = False

    def update(self, arm_angle: Optional[float], torso_bend: Optional[float]) -> Tuple[str, int, bool]:
        just_completed = False
        if arm_angle is None or torso_bend is None:
            return (self.state, self.rep_count, just_completed)
        if torso_bend > self.max_torso_angle:
            return ('invalid_posture', self.rep_count, False)
        if self.state == 'invalid_posture':
            if torso_bend <= self.max_torso_angle:
                self.state = 'neutral'
                self.has_swung_forward = False
        if self.state == 'neutral':
            if arm_angle <= self.forward_threshold:
                self.state = 'swinging_forward'
                self.has_swung_forward = True
            elif arm_angle >= self.backward_threshold:
                self.state = 'swinging_backward'
        elif self.state == 'swinging_forward':
            if arm_angle >= self.backward_threshold:
                self.state = 'swinging_backward'
        elif self.state == 'swinging_backward':
            if self.neutral_min <= arm_angle <= self.neutral_max:
                if self.has_swung_forward:
                    self.rep_count += 1
                    just_completed = True
                self.state = 'neutral'
                self.has_swung_forward = False
        return (self.state, self.rep_count, just_completed)

class PendulumExercise(BaseExercise):
    def __init__(self, config, intrinsics):
        super().__init__(config, intrinsics)
        self.detector = PendulumDetector(depth_intrinsics=intrinsics)
        self.counter = PendulumRepCounter()
        self.angle_smoother = EMAFilter(alpha=0.3)

    def process_frame(self, color_image, depth_frame, depth_image):
        result = self.detector.process_frame(color_image, depth_frame, depth_image)
        is_correct = False
        cue_text = 'Step into frame'
        state_text = 'None'
        just_completed = False
        rep_count = self.counter.rep_count
        smoothed_metric = None
        metric_name = 'Arm Angle'
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
        raw_metric = result.get('arm_swing_angle_deg')
        smoothed_metric = self.angle_smoother.update(raw_metric)
        
        if smoothed_metric is not None:
            state_text, rep_count, just_completed = self.counter.update(smoothed_metric, result.get('torso_bend_angle_deg'))
            is_correct = state_text != 'invalid_posture'
            if not is_correct:
                cue_text = 'BEND OVER MORE'
            else:
                cue_text = 'KEEP SWINGING'
                
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
