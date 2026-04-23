import cv2

print(cv2.__version__)

cam = cv2.VideoCapture(0)
while(1):
    ret, frame = cam.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2
    
    cv2.circle(frame, (cx,cy), radius=5, color=(255,0,0), thickness=-1)
    cv2.imshow('frame', frame)
    
cam.release()
cv2.destroyAllWindows()
