"""
testCoordinate.py: coordinate tracking of an object detected by the webcam.

Notes:
    - 4/6/26: Updated matrices and did some bugfixing to get the coordinates to (hypothetically) work.  Moving the mouse left and right changed the x coordinate
    and moving it forward and backwards changed the y coordinate.  Z coordinate does not appear to be significant.  Waiting to test using some inverse kinematics
    algorithm(s) to see if we can use this to get everything to work.
    - 4/8/26 - Limit testing the bounds of the coordinate system
        - X coordinate - goes from -0.33 to 0.28
        - Y coordinate - goes from 0.152 to 0.44

"""

import cv2

import time
#from gpiozero import AngularServo
from detection_ik_target import compute_target_base_from_bbox, pick_best_detection
#servo =AngularServo(18, initial_angle=0, min_pulse_width=0.0006, max_pulse_width=0.0023)

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects: 
                objectInfo.append([box, className, float(confidence)])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    #servo.angle = -90
                    #time.sleep(2)
                    #servo.angle = 90
    
    return img,objectInfo


if __name__ == "__main__":
    # Tune these for your camera mounting + working distance.
    DEPTH_M = 0.60
    FOV_DEG_X = 60.0
    # Default axis alignment (adjust to your camera mount).
    # base_X = +cam_Z, base_Y = -cam_X, base_Z = -cam_Y
    R_CAM_TO_BASE = ( # Rotational measurement of the difference between the motor base and camera AKA the camera is facing 28.5 degrees downward from it's perch
        (1.0, 0.0, 0.0),
        (0.0, -0.879, 0.478), # Measurement of 28.5 degrees was calculated by placing a dot in the center of the viewport and then measuring the distance 
        (0.0, -0.478, -0.879),# from that to the base and using trig
    )
    T_CAM_TO_BASE_M = (-0.0254, 0.0127, 0.3810) # Positional measurement of the difference between the camera position and motor base position
    last_print_s = 0.0


    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)
    
    
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2, objects=['banana','book','cup','mouse'])

        best = pick_best_detection(objectInfo)
        if best is not None:
            bbox_xywh, class_name, conf = best
            img_h = float(img.shape[0])
            img_w = float(img.shape[1])

            target = compute_target_base_from_bbox(
                bbox_xywh,
                img_w,
                img_h,
                depth_m=DEPTH_M,
                fov_deg_x=FOV_DEG_X,
                R_cam_to_base=R_CAM_TO_BASE,
                t_cam_to_base_m=T_CAM_TO_BASE_M,
            )

            now_s = time.time()
            if now_s - last_print_s > 0.75:
                last_print_s = now_s
                print(
                    f"Detected {class_name} (conf={conf:.2f}) -> "
                    f"target_base x={target['x']:.3f} y={target['y']:.3f} z={target['z']:.3f} "
                    f"(theta1={target['theta1_deg']:.1f} deg)"
                )
        
        
        
        cv2.imshow("Output",img)
        cv2.waitKey(1)
    
