import math
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
hsv_gradient = np.zeros((50, 640, 3), dtype=np.uint8)
for x in range(640):
    hue = int((x / 640) * 179)
    hsv_gradient[:, x] = (hue, 255, 255)  # H, S, V

# Convert HSV to BGR for display
gradient_bar = cv2.cvtColor(hsv_gradient, cv2.COLOR_HSV2BGR)

# To store previous fingertip position
prev_x, prev_y = 0, 0



def fingers_up(handLms):
    fingers = []
    # Thumb: check x not y
    if handLms.landmark[4].x < handLms.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Fingers: tip.y < pip.y means finger is up
    tips_ids = [8, 12, 16, 20]
    for tip_id in tips_ids:
        if handLms.landmark[tip_id].y < handLms.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip for mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    brush_color=(255,255,0)
    curr_x, curr_y = 0, 0
    img[0:50,0:640]=gradient_bar
    cv2.rectangle(img, (10, 60), (60, 110), brush_color, cv2.FILLED)
    cv2.putText(img, "Color", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Index fingertip
            fingers=fingers_up(handLms)
            h, w, c = img.shape
            lm = handLms.landmark[8]
            tm = handLms.landmark[4]
            index_x, index_y = int(lm.x * w), int(lm.y * h)
            thumb_x, thumb_y = int(tm.x * w), int(tm.y * h)
            distance= math.hypot(index_x-thumb_x, index_y-thumb_y)

            if index_y<50:
                if 0<=index_x<=640:
                    brush_color=gradient_bar[0,index_x].tolist()
            # Only draw if previous point is not (0,0)
            if prev_x != 0 and prev_y != 0:
                if fingers[0]==0 and fingers[1]==1 and fingers[2]==1:
                    cv2.circle(img, (index_x, index_y), 10, (255,0,0), cv2.FILLED)
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 0), 10)
                elif distance<25:
                    cv2.circle(img, (index_x, index_y), 10, brush_color, cv2.FILLED)
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), brush_color, 5)

            # Update previous point
            prev_x, prev_y = index_x, index_y

    else:
        # If hand not detected, reset previous point
        prev_x, prev_y = 0, 0

    # Combine camera frame and drawing canvas
    img = cv2.addWeighted(img, 0.7, canvas, 0.7, 0)

    cv2.imshow("GestureBoard - Drawing Mode", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
