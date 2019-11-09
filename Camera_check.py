import cv2

cap = cv2.VideoCapture(0)
cam = cv2.VideoCapture(1)

while(True):
    ret, img = cap.read()
    cv2.imshow('Built-in',img)
    ret, img = cam.read()
    cv2.imshow('USB',img)
    if cv2.waitKey(1) & 0xFF == ord('o'):
        break
