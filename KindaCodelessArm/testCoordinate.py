"""
testCoordinate.py: coordinate tracking of an object detected by the webcam.

Notes:
    - 4/6/26: Updated matrices and did some bugfixing to get the coordinates to (hypothetically) work.  Moving the mouse left and right changed the x coordinate
    and moving it forward and backwards changed the y coordinate.  Z coordinate does not appear to be significant.  Waiting to test using some inverse kinematics
    algorithm(s) to see if we can use this to get everything to work.
    - 4/8/26 - Limit testing the bounds of the coordinate system using a mouse on the camera in the keck lab
        - X coordinate - goes from -0.33 to 0.28 ===> X-Coordinate seems to be fixed, sits around 0 when mouse is placed directly in front of the arm
        - Y coordinate - goes from 0.152 to 0.44
        - For next time: use empirical measurements to determine how accurate the coordinate system is and whether or not we can fix it by messing with the translation matrix.  
        Use Dondi's triangle method to calculate angles for the motors.  Pan can be calculated by drawing a right triangle along the 2D plane using the x and y coordinate from the camera.
        The hypotenuse of that triangle can become d and we can use the measurements of the arm and forearm segments to invoke law of cosines and get angle measurements for the other motors.
    - 4/12/26 - Tested pointing the shoulder_pan motor towards the object based on conceptual right triangle drawn on the table.  This seemed to work well, but when we tried to calculate the 
    shoulder lift and elbow flex, it was reaching far past the object.  Could be some issues with the way we're approaching the triangle.  See trig calculations in main loop for more info.
    
    For next time: tune triangle calculation and figure out what might be wrong with shoulder_lift and elbow_flex.  Also need to tune wrist_flex due to difference in motor degrees and conceptual
    degrees.
"""

import cv2
import math
import time
#from gpiozero import AngularServo
from detection_ik_target import compute_target_base_from_bbox, pick_best_detection
#servo =AngularServo(18, initial_angle=0, min_pulse_width=0.0006, max_pulse_width=0.0023)
from utils.arm_interface import TRIG_MEASUREMENTS

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
    R_CAM_TO_BASE = ( # Rotational measurement of the difference between the motor base and camera AKA the camera is facing 28.5 degrees downward from it's perch
        (1.0, 0.0, 0.0),
        (0.0, -0.879, 0.478), # Measurement of 28.5 degrees was calculated by placing a dot in the center of the viewport and then measuring the distance 
        (0.0, -0.478, -0.879),# from that to the base and using trig
    )
    T_CAM_TO_BASE_M = (-0.127, -0.0254, 0.6810) # Positional measurement of the difference between the camera position and motor base position - y was -0.0254
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
                
                # TRIG CALCULATIONS TO GET MOTOR ANGLES 
                
                # Calculate shoulder pan angle
                sp_adjacent = target['y']
                sp_opposite = target['x']
                sp_hypotenuse = math.sqrt((sp_adjacent**2 + sp_opposite**2))
                
                shoulder_pan_angle = math.atan2(sp_opposite, sp_adjacent)
                
                # Calculate shoulder lift angle
                sl_opposite = TRIG_MEASUREMENTS["base_to_tip"] - TRIG_MEASUREMENTS["ground_to_shoulder"]/2
                sl_adjacent = sp_hypotenuse
                sl_hypotenuse = math.sqrt((sl_opposite**2 + sl_adjacent**2))
                
                print(f"Conceptual shoulder lift triangle has lengths: opposite - {sl_opposite}, adjacent - {sl_adjacent}, hypotenuse - {sl_hypotenuse}")
                
                # Calculate elbow lift angle
                el_c = sl_hypotenuse
                el_a = TRIG_MEASUREMENTS["lower_arm"]
                el_b = TRIG_MEASUREMENTS["forearm"]
                
                print(f"Conceptual elbow lift triangle has lengths: a - {el_a}, b - {el_b}, c - {el_c}")
                
                thetaB = math.acos((el_a**2 + el_c**2 - el_b**2)/(2*el_a*el_c))
                shoulder_lift_angle = math.pi - (thetaB + math.atan2(sl_opposite, sl_adjacent) - 0.22) # Subtract 12 degree motor error offset
                
                elbow_lift_angle = math.acos((el_a**2 + el_b**2 - el_c**2)/(2*el_b*el_a)) # Law of cosines to find theta(c)
                
                # Calculate wrist flex angle
                # We can just add theta(a) from elbow lift and the angle from the shoulder lift calulation
                
                sl_part = math.atan2(sl_adjacent, sl_opposite)
                thetaA = math.acos(((el_c**2 + el_b**2 - el_a**2)/(2*el_b*el_c)))
                
                wrist_flex_angle = sl_part + thetaA
                
                
                print(
                    f"Detected {class_name} (conf={conf:.2f}) -> "
                    f"target_base x={target['x']:.3f} y={target['y']:.3f} z={target['z']:.3f} "
                    f"(theta1={target['theta1_deg']:.1f} deg)"
                    f"=> Shoulder pan angle: {shoulder_pan_angle}\nShoulder lift angle: {shoulder_lift_angle}\nElbow lift angle: {elbow_lift_angle}\nWrist flex angle: {wrist_flex_angle}"
                )
        
        
        
        cv2.imshow("Output",img)
        cv2.waitKey(1)
    
