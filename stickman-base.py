import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Pose dan OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Mendapatkan resolusi layar pengguna
screen_width = 640  # Ubah sesuai resolusi layar kamu
screen_height = 480

# Warna untuk stickman (hijau)
STICKMAN_COLOR = (0, 255, 0)

# Membuka webcam
cap = cv2.VideoCapture(0)

# Cek apakah webcam terbuka
if not cap.isOpened():
    print("Error: Webcam tidak terbuka.")
    exit()

# Setup Pose Estimation
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Buat salinan frame untuk tampilan stickman
        stickman_frame = np.zeros_like(frame)

        if not ret:
            print("Error: Gagal membaca frame dari webcam.")
            break

        # Mengubah frame ke RGB untuk proses di MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Deteksi pose
        results = pose.process(image_rgb)

        # Jika pose terdeteksi, gambar stickman
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                h, w, _ = stickman_frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Gambar lingkaran di titik sendi
                cv2.circle(stickman_frame, (cx, cy), 5, STICKMAN_COLOR, -1)

            # Daftar landmark untuk menggambar stickman
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

            # Menggambar stickman
            for line in stickman_lines:
                p1 = results.pose_landmarks.landmark[line[0]]
                p2 = results.pose_landmarks.landmark[line[1]]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)

                cv2.line(stickman_frame, (x1, y1), (x2, y2), STICKMAN_COLOR, 3)

        # Menampilkan jendela pertama: tampilan asli dari kamera
        #cv2.imshow('Camera Feed', frame)

        # Menampilkan jendela kedua: tampilan stickman
        cv2.imshow('Stickman Tracker', stickman_frame)
        cv2.moveWindow('Stickman Tracker', int(screen_width / 1), int(screen_height / 4))


        # Keluar dengan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release dan tutup jendela
cap.release()
cv2.destroyAllWindows()
