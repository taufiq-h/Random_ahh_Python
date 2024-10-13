
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Set up screen settings
screen_height, screen_width = 480, 640  # Standard resolution for testing
bullet_radius = 5
enemy_radius = 20
enemy_bullet_radius = 5
bullet_speed = 5
enemy_speed = 2
enemy_bullet_speed = 3
bullets = []  # List to store bullets from player
enemies = []  # List to store enemies
enemy_bullets = []  # List to store bullets from enemies
score = 0  # Initialize score

# Function to add a new enemy at the top of the screen
def add_enemy():
    enemy_x = random.randint(enemy_radius, screen_width - enemy_radius)
    enemy_y = 0
    enemies.append([enemy_x, enemy_y])

# Function for enemy shooting
def enemy_shoot(enemy_x, enemy_y):
    enemy_bullets.append([enemy_x, enemy_y + enemy_radius])

# Start capturing video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Timer for enemy spawning and shooting
last_enemy_time = time.time()
last_enemy_shoot_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error accessing webcam")
        break

    # Flip frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Convert frame to black for background
    #frame[:] = (0, 0, 0)

    # Convert BGR frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Hand detection and bullet shooting
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of index finger tip for shooting
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_tip.x * screen_width)
            index_finger_y = int(index_finger_tip.y * screen_height)

            # Detect if thumb is open to shoot
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            thumb_open = thumb_tip.x < thumb_mcp.x  # Thumb open detection for left hand

            if thumb_open:
                bullets.append([index_finger_x, index_finger_y])

            # Draw index finger point for reference
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), cv2.FILLED)

    # Move bullets upwards and remove if off-screen
    for bullet in bullets[:]:
        bullet[1] -= bullet_speed
        if bullet[1] < 0:
            bullets.remove(bullet)

    # Add new enemy at intervals
    if time.time() - last_enemy_time > 2:  # Spawn enemy every 2 seconds
        add_enemy()
        last_enemy_time = time.time()

    # Move enemies downwards
    for enemy in enemies[:]:
        enemy[1] += enemy_speed
        if enemy[1] > screen_height:
            enemies.remove(enemy)  # Remove enemy if it goes off-screen

        # Check collision with bullets
        for bullet in bullets[:]:
            distance = np.sqrt((enemy[0] - bullet[0])**2 + (enemy[1] - bullet[1])**2)
            if distance < enemy_radius + bullet_radius:
                enemies.remove(enemy)  # Destroy enemy
                bullets.remove(bullet)  # Remove bullet on impact
                score += 10  # Increase score for destroying enemy
                break  # Exit bullet loop as enemy is destroyed

        # Enemy shoots projectile every 3 seconds
        if time.time() - last_enemy_shoot_time > 3:
            enemy_shoot(enemy[0], enemy[1])
            last_enemy_shoot_time = time.time()

    # Move enemy bullets downwards
    for enemy_bullet in enemy_bullets[:]:
        enemy_bullet[1] += enemy_bullet_speed
        # Check collision with player's hand
        if (enemy_bullet[0] >= index_finger_x - 10 and enemy_bullet[0] <= index_finger_x + 10) and \
           (enemy_bullet[1] >= index_finger_y - 10 and enemy_bullet[1] <= index_finger_y + 10):
            enemy_bullets.remove(enemy_bullet)  # Remove enemy bullet on hit
            score -= 5  # Decrease score for getting hit

        if enemy_bullet[1] > screen_height:
            enemy_bullets.remove(enemy_bullet)  # Remove if off-screen

    # Draw bullets on frame
    for bullet_x, bullet_y in bullets:
        cv2.circle(frame, (bullet_x, bullet_y), bullet_radius, (0, 255, 255), -1)

    # Draw enemies on frame
    for enemy_x, enemy_y in enemies:
        cv2.circle(frame, (enemy_x, enemy_y), enemy_radius, (255, 0, 0), -1)

    # Draw enemy bullets on frame
    for enemy_bullet_x, enemy_bullet_y in enemy_bullets:
        cv2.circle(frame, (enemy_bullet_x, enemy_bullet_y), enemy_bullet_radius, (0, 0, 255), -1)

    # Display score
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Touhou Hand Tracking Game with Enemies", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
