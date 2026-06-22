import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle between three points."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point (Vertex)
    c = np.array(c) # End point
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def analyze_shoulder_movement(landmarks):
    """
    Evaluates Shoulder Abduction/Flexion.
    Note: Because the video is mirrored, MediaPipe's 'RIGHT' 
    landmarks are tracking your physical LEFT side.
    """
    def get_coords(landmark):
        return [landmark.x, landmark.y, landmark.z]

    # Extract coordinates for the arm and torso
    hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    elbow = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    wrist = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # Calculate Shoulder Angle (Hip -> Shoulder -> Elbow)
    # Arm down at side is ~0-20 deg. Arm raised parallel to ground is ~90 deg.
    shoulder_angle = calculate_angle_3d(hip, shoulder, elbow)
    
    # Calculate Elbow Angle (Shoulder -> Elbow -> Wrist)
    # Checks if the user is bending their elbow to compensate.
    elbow_angle = calculate_angle_3d(shoulder, elbow, wrist)

    feedback = []
    color = (255, 255, 255) # Default White for text

    # 1. Form Logic: Ensure the arm is kept relatively straight
    if elbow_angle < 140:
        feedback.append("Form: Keep your arm straight!")
        color = (0, 0, 255) # Red for warning
    else:
        # 2. Range of Motion Logic (Shoulder Elevation)
        if shoulder_angle < 45:
            feedback.append(f"Status: Resting ({int(shoulder_angle)} deg)")
        elif 45 <= shoulder_angle < 90:
            feedback.append(f"Status: Raising... ({int(shoulder_angle)} deg)")
            color = (0, 255, 255) # Yellow
        elif 90 <= shoulder_angle < 150:
            feedback.append(f"Status: Good Elevation! ({int(shoulder_angle)} deg)")
            color = (0, 255, 0) # Green
        elif shoulder_angle >= 150:
            feedback.append(f"Status: Excellent Full Reach! ({int(shoulder_angle)} deg)")
            color = (0, 255, 0) # Green

    return feedback, color

# --- MAIN VIDEO LOOP ---
video_source = 0 # Ensure this is your correct camera index (e.g., OAK-D or webcam)
cap = cv2.VideoCapture(video_source)

# 1. Setup Full Screen Window
window_name = 'Shoulder Tracker (Mirror Mode)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame horizontally (1) to create the mirror effect
    frame = cv2.flip(frame, 1)

    # Process the newly mirrored frame
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # 2. Create a pure black screen with the exact same dimensions as the camera frame
    black_screen = np.zeros(frame.shape, dtype=np.uint8)
    
    try:
        # Check if landmarks exist before trying to process them
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Analyze the shoulder movement
            corrections, text_color = analyze_shoulder_movement(landmarks)
            
            # 3. Draw MediaPipe skeleton ON THE BLACK SCREEN
            mp.solutions.drawing_utils.draw_landmarks(
                black_screen, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Joint colors
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Bone colors
            )
            
            # Display text feedback ON THE BLACK SCREEN
            y_offset = 50
            for text in corrections:
                # Using OpenCV BGR format for colors
                cv2.putText(black_screen, text, (30, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
                y_offset += 50
                
    except Exception as e:
        # If no one is on camera or an error occurs, it just shows the empty black screen
        pass
        
    # Show the black screen window
    cv2.imshow(window_name, black_screen)
    
    # Press 'q' or the 'Esc' key (27) to exit full screen and close
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()