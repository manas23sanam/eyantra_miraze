import json
import numpy as np

from pose_geometry_base import EMAFilter
from squat_angle_detector import SquatAngleDetector, SquatRepCounter
from pendulum_detector import PendulumDetector, PendulumRepCounter
from shoulder_circle_detector import ShoulderCircleDetector, ShoulderCircleRepCounter
from sit_to_stand_detector_v2 import SitToStandDetectorV2, SitToStandRepCounter
from upper_trapezius_detector import UpperTrapeziusDetector, UpperTrapRepCounter

def load_config():
    with open('exercises_config.json', 'r') as f:
        return json.load(f)

def select_exercise(config_data):
    print("\n" + "="*30)
    print("  SMART MIRROR EXERCISE TRACKER")
    print("="*30)
    keys = [k for k in config_data.keys() if k != "front_squat"]
    for i, key in enumerate(keys):
        print(f"[{i + 1}] {config_data[key]['name']}")
    
    while True:
        try:
            choice = int(input(f"\nSelect an exercise [1-{len(keys)}]: "))
            if 1 <= choice <= len(keys):
                selected_key = keys[choice - 1]
                return selected_key, config_data[selected_key]
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_exercise_components(key, intrinsics, config):
    if key == "mini_squat" or key == "full_squat":
        detector = SquatAngleDetector(depth_intrinsics=intrinsics)
        counter = SquatRepCounter(
            stand_threshold=config.get("stand_threshold", 160.0),
            squat_threshold=config.get("squat_threshold", 90.0),
            angle_min=config.get("angle_min", 60.0),
            max_angle_diff=config.get("max_angle_diff"),
            max_knee_dist_mm=config.get("max_knee_dist_mm"),
            max_torso_lean=config.get("max_torso_lean")
        )
        return detector, counter
    elif key == "pendulum":
        return PendulumDetector(depth_intrinsics=intrinsics), PendulumRepCounter()
    elif key == "shoulder_circle":
        return ShoulderCircleDetector(depth_intrinsics=intrinsics), ShoulderCircleRepCounter()
    elif key == "sit_to_stand":
        return SitToStandDetectorV2(depth_intrinsics=intrinsics), SitToStandRepCounter()
    elif key == "upper_trapezius":
        return UpperTrapeziusDetector(depth_intrinsics=intrinsics), UpperTrapRepCounter()
    else:
        raise ValueError(f"Unknown exercise {key}")

def process_frame_logic(result, rep_counter, angle_smoother, key, config):
    is_correct = False
    cue_text = "Step into frame"
    state_text = "None"
    rep_count = rep_counter.rep_count
    
    smoothed_metric = None
    metric_name = "Value"

    if not result:
        return None, "Value", False, cue_text, state_text, rep_count

    if key == "mini_squat" or key == "full_squat":
        use_ratio = config.get("use_ratio_mode", False)
        raw_metric = result["vertical_ratio"] if use_ratio else result["knee_angle_deg"]
        smoothed_metric = angle_smoother.update(raw_metric)
        metric_name = "Ratio" if use_ratio else "Angle"
        
        if smoothed_metric is not None:
            state_text, rep_count, _ = rep_counter.update(
                smoothed_metric, result.get("angle_diff"), result.get("knee_dist_mm"), result.get("torso_lean_deg")
            )
            is_correct = True
            
            if state_text == "standing": cue_text = "READY"
            elif state_text == "squatting":
                if smoothed_metric > config.get("angle_max", 100): cue_text = "GO LOWER"
                elif smoothed_metric < config.get("angle_min", 60): cue_text = "TOO DEEP"
                else: cue_text = "GOOD DEPTH"
            elif state_text.startswith("invalid"):
                cue_text = f"{state_text} - STAND UP TO RESET"
                is_correct = False
                
    elif key == "pendulum":
        raw_metric = result.get("arm_swing_angle_deg")
        smoothed_metric = angle_smoother.update(raw_metric)
        metric_name = "Arm Angle"
        
        if smoothed_metric is not None:
            state_text, rep_count, _ = rep_counter.update(smoothed_metric, result.get("torso_bend_angle_deg"))
            is_correct = state_text != "invalid_posture"
            if not is_correct:
                cue_text = "BEND OVER MORE"
            else:
                cue_text = "KEEP SWINGING"
                
    elif key == "shoulder_circle":
        smoothed_metric = result.get("angle_deg")
        metric_name = "Angle"
        if smoothed_metric is not None:
            state_text, rep_count, _ = rep_counter.update(smoothed_metric)
            is_correct = True
            cue_text = "KEEP ROTATING"
            
    elif key == "sit_to_stand":
        raw_metric = result.get("leg_angle_deg")
        smoothed_metric = angle_smoother.update(raw_metric)
        metric_name = "Leg Angle"
        if smoothed_metric is not None:
            state_text, rep_count, _ = rep_counter.update(
                smoothed_metric, 
                result.get("back_angle_deg"),
                result.get("arm_angle_deg")
            )
            is_correct = not state_text.startswith("invalid")
            
            if state_text == "invalid_arms":
                cue_text = "KEEP ARMS STRAIGHT"
            elif state_text == "invalid_back_posture":
                cue_text = "KEEP BACK STRAIGHT"
            else:
                cue_text = "SIT OR STAND"
            
    elif key == "upper_trapezius":
        raw_metric = result.get("neck_tilt_deg")
        smoothed_metric = angle_smoother.update(raw_metric)
        metric_name = "Neck Tilt"
        if smoothed_metric is not None:
            state_text, rep_count, _ = rep_counter.update(smoothed_metric, result.get("tilt_direction"))
            is_correct = True
            if state_text == "upright": cue_text = "STRETCH NECK"
            else: cue_text = "GOOD STRETCH"

    return smoothed_metric, metric_name, is_correct, cue_text, state_text, rep_count


class SessionTracker:
    def __init__(self):
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.mistake_counts = {}
        self.in_error_state = False

    def update(self, state_text, cue_text, just_completed):
        if just_completed:
            self.correct_reps += 1
            
        if state_text and state_text.lower().startswith("invalid"):
            if not self.in_error_state:
                self.incorrect_reps += 1
                self.in_error_state = True
                
                # Clean up the reason (e.g. "KNEES TOO WIDE - STAND UP TO RESET" -> "KNEES TOO WIDE")
                mistake_reason = cue_text.split("-")[0].strip()
                self.mistake_counts[mistake_reason] = self.mistake_counts.get(mistake_reason, 0) + 1
        else:
            self.in_error_state = False
            
    def print_summary(self):
        print("\n" + "="*40)
        print("         WORKOUT SUMMARY")
        print("="*40)
        total_attempts = self.correct_reps + self.incorrect_reps
        accuracy = (self.correct_reps / total_attempts * 100) if total_attempts > 0 else 0.0
        
        print(f"Total Reps Completed: {self.correct_reps}")
        print(f"Total Mistakes Made:  {self.incorrect_reps}")
        print(f"Overall Accuracy:     {accuracy:.1f}%\n")
        
        if self.incorrect_reps > 0:
            print("Areas for Improvement:")
            for reason, count in sorted(self.mistake_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason} ({count} times)")
        else:
            if self.correct_reps > 0:
                print("Perfect form! Keep up the great work!")
            else:
                print("No reps attempted.")
        print("="*40 + "\n")
