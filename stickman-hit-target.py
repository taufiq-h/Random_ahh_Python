import cv2
import mediapipe as mp
import numpy as np
import random
import pygame

# Mendapatkan resolusi layar pengguna
screen_width = 640  # Ubah sesuai resolusi layar kamu
screen_height = 480

#cap.set(cv2.CAP_PROP_FPS, 60)

# Inisialisasi Pygame untuk audio
pygame.mixer.init()
pygame.mixer.music.load('../myth_roid_straight_bet.mp3')  # Ganti dengan path lagu Anda
pygame.mixer.music.play(-1)  # Memutar lagu secara berulang

# Inisialisasi MediaPipe Pose dan OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# Warna untuk stickman dan target (hijau dan merah)
STICKMAN_COLOR = (0, 255, 0)
TARGET_COLOR = (0, 0, 255)

# Inisialisasi variabel
score = 0
target_radius = 30
target_position = [random.randint(target_radius, 640 - target_radius), random.randint(0, 480)]  # Posisi awal target

# Membuka webcam
cap = cv2.VideoCapture(0)

# Setup Pose Estimation
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        # Mengubah frame ke RGB untuk proses di MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Deteksi pose
        results = pose.process(image_rgb)

        # Gambar stickman
        stickman_frame = np.zeros_like(frame)

        if results.pose_landmarks:
            h, w, _ = stickman_frame.shape
            
            # Gambar titik-titik dari landmark
            for landmark in results.pose_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(stickman_frame, (cx, cy), 5, STICKMAN_COLOR, -1)

            # Gambar garis antar landmark
            stickman_lines = [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            ]

            for line in stickman_lines:
                p1 = results.pose_landmarks.landmark[line[0]]
                p2 = results.pose_landmarks.landmark[line[1]]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(stickman_frame, (x1, y1), (x2, y2), STICKMAN_COLOR, 3)

            # Mendapatkan posisi tangan kiri dan tangan kanan
            left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hand_x, left_hand_y = int(left_hand.x * w), int(left_hand.y * h)
            right_hand_x, right_hand_y = int(right_hand.x * w), int(right_hand.y * h)

            # Gambar target pada frame stickman
            cv2.circle(stickman_frame, (target_position[0], target_position[1]), target_radius, TARGET_COLOR, -1)

            # Cek hit untuk kedua tangan
            distance_to_target_left = np.sqrt((left_hand_x - target_position[0]) ** 2 + (left_hand_y - target_position[1]) ** 2)
            distance_to_target_right = np.sqrt((right_hand_x - target_position[0]) ** 2 + (right_hand_y - target_position[1]) ** 2)

            if distance_to_target_left < target_radius or distance_to_target_right < target_radius:
                score += 1
                # Reset target ke posisi acak
                target_position = [random.randint(target_radius, 640 - target_radius), random.randint(0, 480)]

            # Tampilkan score
            cv2.putText(stickman_frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Menggunakan mirror effect
        stickman_frame = cv2.flip(stickman_frame, 1)

        # Menampilkan jendela
        cv2.imshow('Stickman Tracker', stickman_frame)
        cv2.moveWindow('Stickman Tracker', int(screen_width / 1), int(screen_height / 4))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release dan tutup jendela
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()  # Berhenti memutar musik
