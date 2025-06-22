import cv2
import mediapipe as mp

mphands=mp.solutions.hands
hands=mphands.Hands()
drawing=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
while True:
    success, img=cap.read()
    img=cv2.resize(img,(1000,700))
    img=cv2.flip(img,1)
    imgrgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgrgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            drawing.draw_landmarks(imgrgb,handLms,mphands.HAND_CONNECTIONS)

            index_finger_tip = handLms.landmark[8]
            h, w, c = imgrgb.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            cv2.circle(imgrgb, (cx, cy), 15, (0, 0, 153),3)
            cv2.putText(imgrgb, f'{cx}, {cy}', (cx+20, cy-20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("GestureBoard",cv2.cvtColor(imgrgb, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()