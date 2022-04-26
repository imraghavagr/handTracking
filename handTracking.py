from locale import currency
from unittest import result
import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)

#lets initialize mediapipe hands object

mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#for tracking fps
prevTime = 0
currTime = 0
while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    #check if we have multiple hands 
    if results.multi_hand_landmarks:
        for handLms in results. multi_hand_landmarks: #for each hand
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(id,cx,cy)
            #mpDraw.draw_landmarks(img,handLms) #for only drawing dots on landmarks
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) #for drawing dots on landmarks and connections between them
    
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)