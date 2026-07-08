import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle_3d(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def analyze_shoulder_movement(landmarks):

    def get_coords(landmark):
        return [landmark.x, landmark.y, landmark.z]
    hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    elbow = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    wrist = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    shoulder_angle = calculate_angle_3d(hip, shoulder, elbow)
    elbow_angle = calculate_angle_3d(shoulder, elbow, wrist)
    feedback = []
    color = (255, 255, 255)
    if elbow_angle < 140:
        feedback.append('Form: Keep your arm straight!')
        color = (0, 0, 255)
    elif shoulder_angle < 45:
        feedback.append(f'Status: Resting ({int(shoulder_angle)} deg)')
    elif 45 <= shoulder_angle < 90:
        feedback.append(f'Status: Raising... ({int(shoulder_angle)} deg)')
        color = (0, 255, 255)
    elif 90 <= shoulder_angle < 150:
        feedback.append(f'Status: Good Elevation! ({int(shoulder_angle)} deg)')
        color = (0, 255, 0)
    elif shoulder_angle >= 150:
        feedback.append(f'Status: Excellent Full Reach! ({int(shoulder_angle)} deg)')
        color = (0, 255, 0)
    return (feedback, color)
video_source = 0
cap = cv2.VideoCapture(video_source)
window_name = 'Shoulder Tracker (Mirror Mode)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    black_screen = np.zeros(frame.shape, dtype=np.uint8)
    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            corrections, text_color = analyze_shoulder_movement(landmarks)
            mp.solutions.drawing_utils.draw_landmarks(black_screen, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            y_offset = 50
            for text in corrections:
                cv2.putText(black_screen, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
                y_offset += 50
    except Exception as e:
        pass
    cv2.imshow(window_name, black_screen)
    key = cv2.waitKey(10) & 255
    if key == ord('q') or key == 27:
        break
cap.release()
cv2.destroyAllWindows()
