import os
import sys
import glob
import argparse
import csv
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from squat_angle_detector import SquatAngleDetector
from sit_to_stand_detector_v2 import SitToStandDetectorV2, SitToStandChecklist
from shoulder_circle_detector import ShoulderCircleDetector, ShoulderCircleTracker
from pendulum_detector import PendulumDetector, PendulumSwingTracker
from upper_trapezius_detector import UpperTrapeziusDetector

class CustomUpperTrapeziusTracker:

    def __init__(self, target_hold_time=3.0, tilt_threshold=20.0, max_torso_lean=15.0):
        self.target_hold_time = target_hold_time
        self.tilt_threshold = tilt_threshold
        self.max_torso_lean = max_torso_lean
        self.hold_start_time = None
        self.rep_count = 0
        self.current_hold_duration = 0.0
        self.is_holding = False
        self.rep_completed_this_hold = False

    def update(self, neck_tilt_deg, torso_lean_deg, current_time):
        is_correct_posture = neck_tilt_deg is not None and neck_tilt_deg >= self.tilt_threshold and (torso_lean_deg is not None) and (torso_lean_deg < self.max_torso_lean)
        if not is_correct_posture:
            self.hold_start_time = None
            self.current_hold_duration = 0.0
            self.is_holding = False
            self.rep_completed_this_hold = False
            return (self.rep_count, self.current_hold_duration)
        if self.rep_completed_this_hold:
            return (self.rep_count, self.target_hold_time)
        if not self.is_holding:
            self.hold_start_time = current_time
            self.is_holding = True
            self.current_hold_duration = 0.0
        else:
            self.current_hold_duration = current_time - self.hold_start_time
            if self.current_hold_duration >= self.target_hold_time:
                self.rep_count += 1
                self.rep_completed_this_hold = True
                self.current_hold_duration = self.target_hold_time
        return (self.rep_count, self.current_hold_duration)

class CustomPendulumSwingTracker(PendulumSwingTracker):

    def __init__(self, window_size=30, min_range_deg=20.0):
        super().__init__(window_size, min_range_deg)
        self.cycle_count = 0
        self.state = 'neutral'
        self.prev_angle = None

    def update(self, arm_swing_angle):
        status = super().update(arm_swing_angle)
        is_swinging = status['is_swinging']
        if arm_swing_angle is not None and self.prev_angle is not None:
            if len(self.history) >= 10:
                midpoint = sum(self.history) / len(self.history)
                if is_swinging:
                    if self.prev_angle < midpoint <= arm_swing_angle:
                        if self.state == 'backward':
                            self.cycle_count += 1
                        self.state = 'forward'
                    elif self.prev_angle > midpoint >= arm_swing_angle:
                        self.state = 'backward'
        if arm_swing_angle is not None:
            self.prev_angle = arm_swing_angle
        status['cycle_count'] = self.cycle_count
        return status

class SitToStandRepCounter:

    def __init__(self, stand_threshold=160.0, sit_threshold=110.0):
        self.stand_threshold = stand_threshold
        self.sit_threshold = sit_threshold
        self.state = 'standing'
        self.rep_count = 0

    def update(self, leg_angle):
        if leg_angle is None:
            return self.rep_count
        if self.state == 'standing' and leg_angle <= self.sit_threshold:
            self.state = 'sitting'
        elif self.state == 'sitting' and leg_angle >= self.stand_threshold:
            self.state = 'standing'
            self.rep_count += 1
        return self.rep_count

class MiniSquatRepCounter:

    def __init__(self, stand_angle=170.0, squat_zone=(130.0, 170.0)):
        self.stand_angle = stand_angle
        self.squat_min, self.squat_max = squat_zone
        self.state = 'standing'
        self.rep_count = 0
        self.reached_correct_zone_this_rep = False

    def update(self, angle):
        if angle is None:
            return self.rep_count
        if self.state == 'standing':
            if angle <= self.squat_max:
                self.state = 'descending'
                self.reached_correct_zone_this_rep = False
        if self.state in ('descending', 'bottom'):
            if self.squat_min <= angle <= self.squat_max:
                self.reached_correct_zone_this_rep = True
                self.state = 'bottom'
        if self.state in ('descending', 'bottom') and angle >= self.stand_angle:
            if self.reached_correct_zone_this_rep:
                self.rep_count += 1
            self.state = 'standing'
        return self.rep_count

def open_db3(file_path):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_path, repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    return (pipeline, align, intrinsics)

def get_exercise_type(filename):
    fn = filename.lower()
    if 'shoulder circle' in fn or 'shoulder circlr' in fn:
        return 'shoulder_circle'
    elif 'upper trapezius' in fn:
        return 'upper_trapezius'
    elif 'pendulum' in fn:
        return 'pendulum'
    elif 'sit to stand' in fn or 'sit_to_stand' in fn:
        return 'sit_to_stand'
    elif 'mini squat' in fn or 'mini_squat' in fn:
        return 'mini_squat'
    return None

def draw_skeleton_with_feedback(image, landmarks_proto, is_correct):
    color = (0, 200, 0) if is_correct else (0, 0, 230)
    landmark_style = mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=4, circle_radius=4)
    connection_style = mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=3)
    mp.solutions.drawing_utils.draw_landmarks(image, landmarks_proto, mp.solutions.pose.POSE_CONNECTIONS, landmark_drawing_spec=landmark_style, connection_drawing_spec=connection_style)

def main():
    parser = argparse.ArgumentParser(description='Automated DB3 Verification Script')
    parser.add_argument('--show_ui', action='store_true', help='Display visual playback UI')
    parser.add_argument('--save_csv', action='store_true', help='Save summary results to CSV')
    args = parser.parse_args()
    db3_dir = 'C:\\Users\\BIT\\OneDrive\\Desktop\\software mirror\\new'
    db3_files = sorted(glob.glob(os.path.join(db3_dir, '*.db3')))
    if not db3_files:
        print(f'No .db3 files found in {db3_dir}')
        sys.exit(1)
    results_table = []
    all_passed = True
    print('\n' + '=' * 100)
    print(f"{'File':<40} | {'Exercise':<16} | {'Expected':<12} | {'Detected':<12} | {'Reps':<5} | {'Coverage':<8} | {'Frame Rate':<10} | {'Result':<6}")
    print('=' * 100)
    for file_path in db3_files:
        filename = os.path.basename(file_path)
        exercise = get_exercise_type(filename)
        if exercise is None:
            continue
        is_wrong_in_name = 'wrong' in filename.lower() or 'incorrect' in filename.lower()
        expected = 'Wrong Form' if is_wrong_in_name else 'Correct Form'
        pipeline, align, intrinsics = open_db3(file_path)
        detector = None
        tracker = None
        sit_to_stand_checklist = None
        if exercise == 'shoulder_circle':
            detector = ShoulderCircleDetector(depth_intrinsics=intrinsics)
            tracker = ShoulderCircleTracker()
        elif exercise == 'upper_trapezius':
            detector = UpperTrapeziusDetector(depth_intrinsics=intrinsics)
            tracker = CustomUpperTrapeziusTracker()
        elif exercise == 'pendulum':
            detector = PendulumDetector(depth_intrinsics=intrinsics)
            tracker = CustomPendulumSwingTracker(min_range_deg=20.0)
        elif exercise == 'sit_to_stand':
            detector = SitToStandDetectorV2(depth_intrinsics=intrinsics)
            tracker = SitToStandRepCounter()
            sit_to_stand_checklist = SitToStandChecklist(leg_stand_threshold=160.0, leg_sit_threshold=110.0, back_min_angle=150.0, arm_raise_min=70.0, arm_raise_max=110.0)
        elif exercise == 'mini_squat':
            detector = SquatAngleDetector(depth_intrinsics=intrinsics)
            tracker = MiniSquatRepCounter(stand_angle=170.0, squat_zone=(130.0, 170.0))
        total_processed_frames = 0
        total_detected_frames = 0
        correct_frames = 0
        detected_reps = 0
        window_name = f'Playback: {filename}'
        if args.show_ui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    break
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                total_processed_frames += 1
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                timestamp_sec = color_frame.get_timestamp() / 1000.0
                is_correct_frame = False
                landmarks = None
                if exercise == 'shoulder_circle':
                    angle, landmarks = detector.get_shoulder_elbow_angle(color_image, depth_frame, depth_image, side='auto')
                    if landmarks is not None:
                        total_detected_frames += 1
                        status = tracker.update(angle)
                        is_correct_frame = status['reversal_count'] <= 1
                        if is_correct_frame:
                            correct_frames += 1
                        if status['is_complete']:
                            detected_reps += 1
                            tracker.reset()
                elif exercise == 'upper_trapezius':
                    result = detector.process_frame(color_image, depth_frame, depth_image)
                    if result is not None:
                        total_detected_frames += 1
                        landmarks = result['landmarks']
                        neck_tilt = result['neck_tilt_deg']
                        torso_lean = result['torso_lean_deg']
                        detected_reps, _ = tracker.update(neck_tilt, torso_lean, timestamp_sec)
                        is_correct_frame = neck_tilt >= 20.0 and torso_lean < 15.0
                        if is_correct_frame:
                            correct_frames += 1
                elif exercise == 'pendulum':
                    result = detector.process_frame(color_image, depth_frame, depth_image, swinging_side='auto')
                    if result is not None:
                        total_detected_frames += 1
                        landmarks = result['landmarks']
                        torso_bend = result['torso_bend_angle_deg']
                        arm_swing = result['arm_swing_angle_deg']
                        status = tracker.update(arm_swing)
                        is_swinging = status['is_swinging']
                        detected_reps = status['cycle_count']
                        is_correct_frame = torso_bend is not None and 100.0 <= torso_bend <= 145.0 and is_swinging
                        if is_correct_frame:
                            correct_frames += 1
                elif exercise == 'sit_to_stand':
                    result = detector.process_frame(color_image, depth_frame, depth_image, side='auto')
                    if result is not None:
                        total_detected_frames += 1
                        landmarks = result['landmarks']
                        leg_angle = result['leg_angle_deg']
                        detected_reps = tracker.update(leg_angle)
                        checklist_res = sit_to_stand_checklist.evaluate(result)
                        total_checks = checklist_res.get('total_measured', 0)
                        passed_checks = checklist_res.get('passed', 0)
                        is_correct_frame = total_checks > 0 and passed_checks >= total_checks / 2.0
                        if is_correct_frame:
                            correct_frames += 1
                elif exercise == 'mini_squat':
                    result = detector.process_frame(color_image, depth_frame, depth_image, side='auto')
                    if result is not None:
                        total_detected_frames += 1
                        landmarks = result['landmarks']
                        knee_angle = result['knee_angle_deg']
                        detected_reps = tracker.update(knee_angle)
                        is_correct_frame = 130.0 <= knee_angle <= 170.0
                        if result.get('angle_diff') is not None and result['angle_diff'] > 25.0:
                            is_correct_frame = False
                        if result.get('knee_dist_mm') is not None and result['knee_dist_mm'] > 500.0:
                            is_correct_frame = False
                        if is_correct_frame:
                            correct_frames += 1
                if args.show_ui:
                    display_image = color_image.copy()
                    if landmarks is not None:
                        draw_skeleton_with_feedback(display_image, landmarks, is_correct_frame)
                    h, w = display_image.shape[:2]
                    cv2.putText(display_image, f'{exercise.upper()} | Reps: {detected_reps}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    status_text = 'CORRECT' if is_correct_frame else 'WRONG'
                    status_color = (0, 200, 0) if is_correct_frame else (0, 0, 230)
                    cv2.putText(display_image, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
                    cv2.imshow(window_name, display_image)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
        finally:
            pipeline.stop()
            if detector is not None:
                detector.close()
            if args.show_ui:
                cv2.destroyWindow(window_name)
        coverage_pct = total_detected_frames / total_processed_frames * 100 if total_processed_frames > 0 else 0.0
        correct_frame_rate_pct = correct_frames / total_detected_frames * 100 if total_detected_frames > 0 else 0.0
        detected_label = 'Correct Form' if detected_reps >= 1 and correct_frame_rate_pct >= 50.0 else 'Wrong Form'
        result_pass = 'PASS' if expected == detected_label else 'FAIL'
        if result_pass == 'FAIL':
            all_passed = False
        results_table.append({'File': filename, 'Exercise': exercise, 'Expected': expected, 'Detected': detected_label, 'Reps': detected_reps, 'Coverage': f'{coverage_pct:.1f}%', 'Frame rate': f'{correct_frame_rate_pct:.1f}%', 'Result': result_pass})
        print(f'{filename[:40]:<40} | {exercise:<16} | {expected:<12} | {detected_label:<12} | {detected_reps:<5} | {coverage_pct:>6.1f}% | {correct_frame_rate_pct:>8.1f}% | {result_pass:<6}')
    print('=' * 100)
    if args.save_csv:
        csv_path = os.path.join(os.path.dirname(db3_dir), 'code', 'verification_results.csv')
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['File', 'Exercise', 'Expected', 'Detected', 'Reps', 'Coverage', 'Frame rate', 'Result'])
                writer.writeheader()
                writer.writerows(results_table)
            print(f'\nSummary results saved to: {csv_path}')
        except Exception as e:
            print(f'Error saving CSV: {e}')
    if all_passed:
        print('\nALL Expected Matches Detected! Verification PASSED.')
        sys.exit(0)
    else:
        print('\nSome exercises failed verification. Verification FAILED.')
        sys.exit(1)
if __name__ == '__main__':
    main()
