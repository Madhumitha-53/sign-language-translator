import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera Test', frame)
        cv2.waitKey(0)
    else:
        print("Failed to grab frame")
else:
    print("Cannot open camera")

cap.release()
cv2.destroyAllWindows()