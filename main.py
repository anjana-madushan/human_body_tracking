import cv2 as cv
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap  = cv.VideoCapture("D:\SLIIT\Academic\YEAR 04\Research\Data\OneDrive_1_3-19-2024\\pull.mp4")

while True:
  ret, img = cap.read()
  img = cv.resize(img, (600, 400))

  results = pose.process(img)
  mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                         connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))
  cv.imshow("Post Estiation", img)

  h, w, c = img.shape
  opImg = np.zeros([h, w, c])
  opImg.fill(255)
  mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                         connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))
  cv.imshow('Extraction Pose', opImg)

  print(results.pose_landmarks)
  cv.waitKey(1)