import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hand_detector = mp_hands.Hands()

# Initialize variables
x_index, y_index, x_thumb, y_thumb, x_middle, y_middle = 0, 0, 0, 0, 0, 0
last_click_time = 0  # To manage click debounce
click_threshold = 0.3  # Minimum time between clicks (in seconds)
mode = "mouse_mode"  # Current mode: 'mouse_mode' or 'volume_mode'

# Volume control threshold
VOLUME_DISTANCE_THRESHOLD = 50

# Set up webcam capture
camera = cv2.VideoCapture(0)

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

try:
    while True:
        # Capture video frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Process the frame
        frame_height, frame_width, _ = frame.shape
        frame = cv2.flip(frame, 1)  # Mirror the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hand_detector.process(rgb_frame)
        hand_landmarks = results.multi_hand_landmarks

        if hand_landmarks:
            # Process the first detected hand only
            hand_landmark = hand_landmarks[0]
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Extract important landmarks
            for id, landmark in enumerate(hand_landmark.landmark):
                x_pixel = int(landmark.x * frame_width)
                y_pixel = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip
                    x_index, y_index = x_pixel, y_pixel
                    mouse_x = int(x_pixel * screen_width / frame_width)
                    mouse_y = int(y_pixel * screen_height / frame_height)

                    if mode == "mouse_mode":
                        pyautogui.moveTo(mouse_x, mouse_y)
                    cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)

                elif id == 4:  # Thumb tip
                    x_thumb, y_thumb = x_pixel, y_pixel
                    cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 0, 255), -1)

                elif id == 12:  # Middle finger tip
                    x_middle, y_middle = x_pixel, y_pixel
                    cv2.circle(frame, (x_pixel, y_pixel), 5, (255, 0, 0), -1)

            # Gesture to switch to volume control mode
            if y_thumb > y_index and y_middle < y_index and y_middle < y_thumb:
                mode = "volume_mode"
                print("Switched to Volume Control Mode.")

            # Gesture to switch to mouse control mode (example: open hand gesture)
            fingers_up = sum(1 for i in [4, 8, 12, 16, 20] if hand_landmark.landmark[i].y < hand_landmark.landmark[i - 2].y)
            if fingers_up == 5:
                mode = "mouse_mode"
                print("Switched to Mouse Control Mode.")

            # Handle specific modes
            if mode == "mouse_mode":
                # Handle mouse functionality
                distance = calculate_distance(x_thumb, y_thumb, x_index, y_index)
                if distance < VOLUME_DISTANCE_THRESHOLD:
                    current_time = time.time()
                    if current_time - last_click_time > click_threshold:
                        pyautogui.click()
                        last_click_time = current_time  # Update last click time

            elif mode == "volume_mode":
                # Handle volume control functionality
                distance = calculate_distance(x_thumb, y_thumb, x_index, y_index)
                cv2.line(frame, (x_index, y_index), (x_thumb, y_thumb), (0, 255, 0), thickness=3)

                if distance < VOLUME_DISTANCE_THRESHOLD:
                    pyautogui.press("volumedown")
                else:
                    pyautogui.press("volumeup")

        # Display the frame with the current mode
        status_label = f"MODE: {mode.upper()}"
        color = (0, 255, 0) if mode == "mouse_mode" else (0, 0, 255)
        cv2.putText(frame, status_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Hand Gesture Control", frame)

        # Exit loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Release resources
    camera.release()
    cv2.destroyAllWindows()
