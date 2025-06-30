import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Inițializare MediaPipe
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Configurare parametri
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
FRAME_SCALE_FACTOR = 0.8  # Factor de scalare pentru performanță (rezolutie gen)

def initialize_camera():
    """Inițializează camera web și returnează obiectul VideoCapture."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Error: Camera could not be opened.")
        return cap
    except Exception as e:
        print(f"Error initializing camera: {e}")
        exit(1)

def get_angle(p1, p2, p3):
    """Calculează unghiul între trei puncte (p1-p2-p3)."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip pentru a evita erori numerice
    return np.degrees(angle)

def count_fingers(hand_landmarks, handedness):
    """
    Numără degetele ridicate pe baza unghiurilor între articulații și poziției landmark-urilor.

    Args:
        hand_landmarks: Obiectul cu landmark-urile mâinii.
        handedness: Eticheta mâinii ("Left" sau "Right").

    Returns:
        Lista de 1 și 0 reprezentând degetele ridicate (1 = ridicat, 0 = îndoit).
    """
    tips_ids = [4, 8, 12, 16, 20]  # ID-urile vârfurilor degetelor (thumb, index, middle, ring, pinky)
    pip_ids = [3, 6, 10, 14, 18]   # ID-urile articulațiilor PIP
    mcp_ids = [2, 5, 9, 13, 17]    # ID-urile articulațiilor MCP
    fingers = []

    # Degetul mare: combină verificarea coordonatelor cu unghiul CMC-MCP-IP
    thumb_tip = hand_landmarks.landmark[tips_ids[0]]
    thumb_ip = hand_landmarks.landmark[pip_ids[0]]
    thumb_mcp = hand_landmarks.landmark[mcp_ids[0]]
    index_mcp = hand_landmarks.landmark[mcp_ids[1]]

    # Verificare coordonate pentru degetul mare
    thumb_coord_up = False
    if handedness == "Right":
        if thumb_tip.x < thumb_ip.x and thumb_tip.x < index_mcp.x:
            thumb_coord_up = True
    else:  # Left
        if thumb_tip.x > thumb_ip.x and thumb_tip.x > index_mcp.x:
            thumb_coord_up = True

    # Verificare unghi pentru degetul mare
    thumb_angle = get_angle(hand_landmarks.landmark[1], thumb_mcp, thumb_ip)  # CMC-MCP-IP
    thumb_angle_up = thumb_angle > 120  # Prag pentru degetul mare întins
    fingers.append(1 if thumb_coord_up and thumb_angle_up else 0)

    # Celelalte degete: verifică unghiul între MCP, PIP și TIP
    ANGLE_THRESHOLD = 140  # Prag pentru unghiul care indică un deget întins
    for i in range(1, 5):  # Index, middle, ring, pinky
        angle = get_angle(
            hand_landmarks.landmark[mcp_ids[i]],
            hand_landmarks.landmark[pip_ids[i]],
            hand_landmarks.landmark[tips_ids[i]]
        )
        if angle > ANGLE_THRESHOLD:
            fingers.append(1)  # Deget ridicat
        else:
            fingers.append(0)  # Deget îndoit
    return fingers

def get_face_expression(landmarks: list[landmark_pb2.NormalizedLandmark], image_w: int, image_h: int):
    """
    Analizează landmark-urile feței pentru a determina expresia sprâncenelor și a gurii.

    Args:
        landmarks: Lista de landmark-uri normalizate ale feței.
        image_w: Lățimea imaginii în pixeli.
        image_h: Înălțimea imaginii în pixeli.

    Returns:
        Tuplu (eyebrow_expression, mouth_expression, debug_info) ca șiruri de caractere și informații de debug.
    """
    # Landmark-uri pentru buze, sprâncene și ochi
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[78]
    right_mouth = landmarks[308]
    left_eye_center = landmarks[159]
    right_eye_center = landmarks[386]
    left_eyebrow_peak = landmarks[65]
    right_eyebrow_peak = landmarks[295]
    left_eye_upper = landmarks[159]  # Pleoapa superioară stânga
    left_eye_lower = landmarks[145]  # Pleoapa inferioară stânga
    right_eye_upper = landmarks[386]  # Pleoapa superioară dreapta
    right_eye_lower = landmarks[374]  # Pleoapa inferioară dreapta

    def to_pixel(landmark: landmark_pb2.NormalizedLandmark):
        return np.array([int(landmark.x * image_w), int(landmark.y * image_h)])

    # Conversie în pixeli
    top_lip_pt = to_pixel(top_lip)
    bottom_lip_pt = to_pixel(bottom_lip)
    left_mouth_pt = to_pixel(left_mouth)
    right_mouth_pt = to_pixel(right_mouth)
    left_eye_center_pt = to_pixel(left_eye_center)
    right_eye_center_pt = to_pixel(right_eye_center)
    left_eyebrow_peak_pt = to_pixel(left_eyebrow_peak)
    right_eyebrow_peak_pt = to_pixel(right_eyebrow_peak)
    left_eye_upper_pt = to_pixel(left_eye_upper)
    left_eye_lower_pt = to_pixel(left_eye_lower)
    right_eye_upper_pt = to_pixel(right_eye_upper)
    right_eye_lower_pt = to_pixel(right_eye_lower)

    # Calculează distanțele
    mouth_open = np.linalg.norm(top_lip_pt - bottom_lip_pt)
    mouth_width = np.linalg.norm(left_mouth_pt - right_mouth_pt)
    eye_dist = np.linalg.norm(left_eye_center_pt - right_eye_center_pt)

    # Distanța sprâncenelor față de centrul ochilor
    left_eyebrow_eye_dist = left_eyebrow_peak_pt[1] - left_eye_center_pt[1]
    right_eyebrow_eye_dist = right_eyebrow_peak_pt[1] - right_eye_center_pt[1]
    avg_eyebrow_eye_dist = (left_eyebrow_eye_dist + right_eyebrow_eye_dist) / 2

    # Distanța între pleoape (înălțimea ochilor)
    left_eye_height = np.linalg.norm(left_eye_upper_pt - left_eye_lower_pt)
    right_eye_height = np.linalg.norm(right_eye_upper_pt - right_eye_lower_pt)
    avg_eye_height = (left_eye_height + right_eye_height) / 2

    # Verificare coordonata z pentru sprâncene (pentru robustețe la rotații)
    left_eyebrow_z = landmarks[65].z
    right_eyebrow_z = landmarks[295].z
    avg_eyebrow_z = (left_eyebrow_z + right_eyebrow_z) / 2

    # Praguri pentru expresii
    eyebrow_threshold_high = -0.2 * eye_dist  # Prag mai permisiv pentru surpriză
    eyebrow_threshold_low = -0.15 * eye_dist
    eye_height_threshold = 0.15 * eye_dist  # Prag pentru ochi deschiși
    mouth_open_threshold = 0.2 * eye_dist   # Prag pentru gură deschisă

    # Detecție expresie sprâncene
    eye_mimics = "Neutral"
    if avg_eyebrow_eye_dist < eyebrow_threshold_high and avg_eye_height > eye_height_threshold:
        eye_mimics = "Surprised"
    elif avg_eyebrow_eye_dist > eyebrow_threshold_low:
        eye_mimics = "Angry"

    # Detecție expresie gură
    mouth_mimics = "Neutral"
    if mouth_open > mouth_open_threshold:
        mouth_mimics = "Mouth Open"
    elif mouth_width > eye_dist * 0.75:
        mouth_mimics = "Smiley"
    elif mouth_width < eye_dist * 0.6:
        mouth_mimics = "Sad"

    return eye_mimics, mouth_mimics

def draw_text_with_background(frame, text, position, font_scale=1.0, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              text_color=(255, 0, 128), bg_color=(0, 0, 0, 128)):
    """Desenează text cu fundal semi-transparent pe frame."""
    text_size, _ = cv2.getTextSize(text, font, font_scale, 1)
    x, y = position
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_size[1] - 5), 
                  (x + text_size[0] + 5, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, position, font, font_scale, text_color, 2)

def process_frame(frame, hands, face_detection, face_mesh):
    """Procesează un frame pentru detecția mâinilor și feței."""
    ih, iw, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    face_expression_eyebrows, face_expression_mouth= "Fara detectie", "Fara detectie"
    face_results_mesh = face_mesh.process(rgb)
    if face_results_mesh.multi_face_landmarks:
        face_landmarks_list = face_results_mesh.multi_face_landmarks[0].landmark
        face_expression_eyebrows, face_expression_mouth = get_face_expression(face_landmarks_list, iw, ih)

    results = hands.process(rgb)
    hand_data = []
    if results.multi_hand_landmarks:
        for i, (handLms, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            handedness_label = hand_handedness.classification[0].label
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers_up = count_fingers(handLms, handedness_label)
            numar_degete = sum(fingers_up)

            gesture = "Unknown"
            if fingers_up == [0, 0, 0, 0, 0]:
                gesture = "Fist"
            elif fingers_up == [1, 1, 1, 1, 1]:
                gesture = "Palm"
            elif fingers_up == [0, 1, 1, 0, 0]:
                gesture = "Victory"
            elif fingers_up == [1, 0, 0, 0, 0]:
                gesture = "Like"
            elif fingers_up == [0, 1, 0, 0, 0]:
                gesture = "Look!"
            elif fingers_up == [0, 0, 1, 0, 0]:
                gesture = "F**k Georgescu!"

            hand_data.append((i + 1, numar_degete, gesture))

    return frame, hand_data, face_expression_eyebrows, face_expression_mouth

def main():
    """Funcția principală pentru rularea aplicației."""
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                           min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                      min_detection_confidence=0.5, refine_landmarks=True)

    cap = initialize_camera()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error!: Frame is corrupted.")
                break

            frame = cv2.resize(frame, None, fx=FRAME_SCALE_FACTOR, fy=FRAME_SCALE_FACTOR)
            frame = cv2.flip(frame, 1)

            frame, hand_data, face_expression_eyebrows, face_expression_mouth = process_frame(
                frame, hands, face_detection, face_mesh)

            draw_text_with_background(frame, f'Eyebrows expression: {face_expression_eyebrows}', (10, 30))
            draw_text_with_background(frame, f'Mouth expression: {face_expression_mouth}', (10, 80))
            
            for i, (hand_id, numar_degete, gesture) in enumerate(hand_data):
                y_base = 140 + i * 60
                draw_text_with_background(frame, f'Hand no {hand_id}: {numar_degete} fingers up', (10, y_base))
                draw_text_with_background(frame, f'Gesture: {gesture}', (10, y_base + 30))

            draw_text_with_background(frame, 'For exit press ESC', (10, frame.shape[0] - 20),
                                     text_color=(0, 0, 255))

            cv2.imshow("Face and fingers gesture recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        hands.close()
        face_detection.close()
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()