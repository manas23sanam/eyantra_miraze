import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle between three points."""
    a = np.array(a)
    b = np.array(b) 
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance_3d(a, b):
    """Calculates the 3D Euclidean distance between two landmarks."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def analyze_single_leg_squat(landmarks, baseline_torso):
    """
    Dynamically detects the working leg, evaluates the Squat depth, 
    and checks for back rounding using 3D torso compression.
    """
    def get_coords(landmark):
        return [landmark.x, landmark.y, landmark.z]

    # 1. Get both ankles
    right_ankle = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_ankle = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    working_leg = "NONE"
    shoulder, hip, knee, ankle = None, None, None, None

    # Detect Right Leg
    if right_ankle[1] > left_ankle[1] + 0.05:
        working_leg = "RIGHT"
        shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        knee = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        ankle = right_ankle
        
    # Detect Left Leg
    elif left_ankle[1] > right_ankle[1] + 0.05:
        working_leg = "LEFT"
        shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        hip = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        knee = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        ankle = left_ankle
        
    else:
        return ["Status: Lift one foot to begin!"], (255, 255, 255), 180, "NONE", 0

    # 2. Calculate Angles and Distances
    knee_angle = calculate_angle_3d(hip, knee, ankle)
    hip_angle = calculate_angle_3d(shoulder, hip, knee)
    current_torso_length = calculate_distance_3d(shoulder, hip)

    feedback = []
    color = (0, 255, 0) # Green

    # 3. Depth Logic
    if knee_angle > 160:
        feedback.append(f"Status: Standing ({int(knee_angle)}°)")
    elif 100 <= knee_angle <= 160:
        feedback.append(f"Status: Squatting... ({int(knee_angle)}°)")
    elif knee_angle < 100:
        feedback.append(f"Status: Good Depth! ({int(knee_angle)}°)")
        color = (255, 255, 0) # Cyan

    # 4. Posture & Back Check
    if hip_angle < 75:
        feedback.append("Posture: Torso leaning too far forward!")
        color = (0, 0, 255) # Red
        
    if baseline_torso is not None:
        shrinkage_ratio = current_torso_length / baseline_torso
        if shrinkage_ratio < 0.88: # 12% compression threshold
            feedback.append(f"Back Form: Rounded! (Compressed to {int(shrinkage_ratio * 100)}%)")
            color = (0, 165, 255) # Orange

    return feedback, color, knee_angle, working_leg, current_torso_length

# --- MAIN VIDEO LOOP ---
video_source = r'C:\Users\BIT\OneDrive\Desktop\software mirror\dataset\Single_leg_squat.mp4'
cap = cv2.VideoCapture(video_source)

window_name = 'Single Leg Squat Tracker'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# State Variables
rep_count = 0
squat_stage = "UP" 
baseline_torso_length = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    black_screen = np.zeros(frame.shape, dtype=np.uint8)
    
    # Check if a person is actually detected in the frame
    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Unpack all 5 variables safely
            corrections, text_color, current_knee_angle, active_leg, current_torso = analyze_single_leg_squat(landmarks, baseline_torso_length)
            
            # --- CALIBRATE BASELINE ---
            if current_knee_angle > 170 and baseline_torso_length is None and active_leg != "NONE":
                baseline_torso_length = current_torso
                print(f"Torso Calibrated: {baseline_torso_length}")

            # --- REP COUNTING LOGIC ---
            if active_leg != "NONE":
                if current_knee_angle < 100:
                    if squat_stage == "UP":
                        squat_stage = "DOWN"
                elif current_knee_angle > 160:
                    if squat_stage == "DOWN":
                        squat_stage = "UP"
                        rep_count += 1
            else:
                squat_stage = "UP"
            
            # Draw skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                black_screen, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Display HUD
            cv2.putText(black_screen, f"ACTIVE LEG: {active_leg}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            y_offset = 90
            for text in corrections:
                cv2.putText(black_screen, text, (30, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                y_offset += 30
                
            cv2.putText(black_screen, f"REPS: {rep_count}", (frame.shape[1] - 180, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
        except Exception as e:
            # If the math fails, print exactly why in the terminal
            print(f"MATH ERROR: {e}")
    else:
        # If MediaPipe can't see the body, tell the user on screen
        cv2.putText(black_screen, "Waiting for full body in frame (Need to see ankles)...", (50, frame.shape[0] // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow(window_name, black_screen)
    
    if cv2.waitKey(10) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()