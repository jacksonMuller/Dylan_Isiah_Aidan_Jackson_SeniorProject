import cv2

print(cv2.__version__)

cam = cv2.VideoCapture(0)
while(1):
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()