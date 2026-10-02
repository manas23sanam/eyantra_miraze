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

def analyze_mini_squat(landmarks):

    def get_coords(landmark):
        return [landmark.x, landmark.y, landmark.z]
    shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    knee = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    ankle = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    knee_angle = calculate_angle_3d(hip, knee, ankle)
    hip_angle = calculate_angle_3d(shoulder, hip, knee)
    feedback = []
    color = (0, 255, 0)
    if knee_angle > 165:
        feedback.append(f'Status: Standing ({int(knee_angle)} degrees)')
    elif 145 <= knee_angle <= 165:
        feedback.append(f'Status: Good Mini Squat Depth! ({int(knee_angle)} degrees)')
    elif knee_angle < 145:
        feedback.append(f'Status: Too Deep for Mini Squat! ({int(knee_angle)} degrees)')
        color = (0, 0, 255)
    if hip_angle < 110:
        feedback.append('Posture: Keep your chest up! Leaning too far.')
        color = (0, 0, 255)
    return (feedback, color)
video_source = 0
cap = cv2.VideoCapture(video_source)
window_name = 'Mini Squat Tracker (Mirror Mode)'
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
        landmarks = results.pose_landmarks.landmark
        corrections, text_color = analyze_mini_squat(landmarks)
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
