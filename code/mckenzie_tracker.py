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

def analyze_mckenzie(landmarks):
    """
    Tracks the Prone Press-Up. Auto-detects the side closest to the camera.
    """
    # 1. Auto-Detect the closest side using visibility scores
    right_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
    left_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility

    def get_coords(landmark_index):
        lm = landmarks[landmark_index]
        return [lm.x, lm.y, lm.z]

    active_side = "RIGHT"
    if left_shoulder_vis > right_shoulder_vis:
        active_side = "LEFT"
        shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        elbow = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        wrist = get_coords(mp_pose.PoseLandmark.LEFT_WRIST.value)
        hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP.value)
        knee = get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value)
    else:
        shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
        knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value)

    # 2. Calculate Key Angles
    # Elbow angle determines if arms are locked out (Rep tracking)
    elbow_angle = calculate_angle_3d(shoulder, elbow, wrist)
    
    # Extension angle determines how much the spine is curving
    extension_angle = calculate_angle_3d(shoulder, hip, knee)

    feedback = []
    color = (0, 255, 0) # Green

    # 3. Form Check: Are the hips lifting off the floor?
    # In MediaPipe, Y=0 is the top of the screen, Y=1 is the bottom.
    # If the Hip Y is significantly smaller than the Knee Y, the hips are floating up.
    hip_knee_y_diff = knee[1] - hip[1] 
    
    if hip_knee_y_diff > 0.08: # 0.08 is the strictness threshold. 
        feedback.append("FORM ERROR: Keep your hips on the floor!")
        color = (0, 0, 255) # Red warning
    else:
        # 4. Phase Tracking (Only if hips are grounded)
        if elbow_angle < 100:
            feedback.append(f"Status: Resting Down (Elbow: {int(elbow_angle)}°)")
        elif 100 <= elbow_angle <= 150:
            feedback.append(f"Status: Pressing Up... (Extension: {int(extension_angle)}°)")
        elif elbow_angle > 150:
            feedback.append(f"Status: Max Extension! (Extension: {int(extension_angle)}°)")
            color = (255, 255, 0) # Cyan for full lockout

    return feedback, color, elbow_angle, active_side

# --- MAIN VIDEO LOOP ---
video_source = r'C:\Users\BIT\OneDrive\Desktop\software mirror\dataset\mckenzie-exersice.mp4'
cap = cv2.VideoCapture(video_source)

window_name = 'McKenzie Press-Up Tracker'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# State Variables for Rep Counting
rep_count = 0
exercise_stage = "DOWN" 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    black_screen = np.zeros(frame.shape, dtype=np.uint8)
    
    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Analyze posture
            corrections, text_color, current_elbow_angle, active_side = analyze_mckenzie(landmarks)
            
            # --- REP COUNTING LOGIC ---
            # Rep counts when you go from resting on the floor (elbows bent) to full press up.
            if current_elbow_angle < 90:
                if exercise_stage == "UP":
                    # You only complete the rep when you return to the floor
                    rep_count += 1
                    exercise_stage = "DOWN"
            elif current_elbow_angle > 150:
                if exercise_stage == "DOWN":
                    exercise_stage = "UP"
            
            # Draw skeleton
            mp.solutions.drawing_utils.draw_landmarks(
                black_screen, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Display HUD
            cv2.putText(black_screen, f"TRACKING SIDE: {active_side}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            y_offset = 90
            for text in corrections:
                cv2.putText(black_screen, text, (30, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                y_offset += 30
                
            cv2.putText(black_screen, f"REPS: {rep_count}", (frame.shape[1] - 180, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
        except Exception as e:
            print(f"MATH ERROR: {e}")
    else:
        cv2.putText(black_screen, "Waiting for full body (Need strict side-profile)...", (50, frame.shape[0] // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow(window_name, black_screen)
    
    if cv2.waitKey(10) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()