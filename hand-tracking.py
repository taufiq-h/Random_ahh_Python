import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame for sound
pygame.mixer.init()
pygame.mixer.music.load('../myth_roid_straight_bet.mp3')
pygame.mixer.music.play()

# Circle and Rhythm settings
circle_radius = 50
screen_height, screen_width = 720, 1280  # Ensure this matches your webcam resolution
circles = []
start_time = time.time()

# Add circle for every beat (based on the song's rhythm)
def add_circle():
    circle_x = np.random.randint(circle_radius, screen_width - circle_radius)
    circle_y = np.random.randint(circle_radius, screen_height - circle_radius)
    circles.append([circle_x, circle_y, time.time()])

# Capture webcam video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)  # Set webcam width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)  # Set webcam height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error accessing webcam")
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Convert back to BGR for OpenCV display
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Draw detected hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            
            # Ensure coordinates scale properly with actual frame size
            index_finger_x = int(index_finger_tip.x * screen_width)  # Scale to screen width
            index_finger_y = int(index_finger_tip.y * screen_height)  # Scale to screen height

            # Check for circle touch
            for circle in circles[:]:
                circle_x, circle_y, circle_spawn_time = circle
                if (index_finger_x - circle_x) ** 2 + (index_finger_y - circle_y) ** 2 < circle_radius ** 2:
                    print(f"Hit! at ({circle_x}, {circle_y})")
                    circles.remove(circle)

            # Draw index finger point for reference
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), cv2.FILLED)

    # Draw and remove circles over time
    current_time = time.time()
    for circle in circles[:]:
        circle_x, circle_y, circle_spawn_time = circle
        if current_time - circle_spawn_time > 3:  # Circles disappear after 3 seconds
            circles.remove(circle)
        else:
            # Draw circles
            cv2.circle(frame, (circle_x, circle_y), circle_radius, (0, 0, 255), 5)

    # Add a circle at regular intervals (based on song beat, adjust as needed)
    if current_time - start_time > 1:  # 1 second per beat (adjust based on song)
        add_circle()
        start_time = current_time

    # Display the resulting frame
    cv2.imshow("osu! with hand tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
