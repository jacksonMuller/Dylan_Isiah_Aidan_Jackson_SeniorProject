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
    - 4/20/26 - Combined testCoordinate.py with the full position function of shoulder_pan_angle.py.  Found that it can be super accurate (usually when the object is near the middle of 
    the camera's view), but the x-coordinate scales very strangely as you move left and right.
    
    
    For next time: Revamp keyboard demo to use a conceptual coordinate and employ the inverse kinematics system from this file to be able to move around.  Possibly keep working on tuning
    camera control, but it may be hardware restrained or too difficult for the next two weeks.  If we can't figure it out, we can always just demonstrate both the accurate and innaccurate
    ranges and talking about the struggles we faced during debugging.
"""

import cv2
import math
import time
#from gpiozero import AngularServo
from detection_ik_target import compute_target_base_from_bbox, pick_best_detection
#servo =AngularServo(18, initial_angle=0, min_pulse_width=0.0006, max_pulse_width=0.0023)
from utils.arm_interface import TRIG_MEASUREMENTS, RobotMotorInterface, DEFAULT_MOTORS_DEGREES

arm = RobotMotorInterface(motors=DEFAULT_MOTORS_DEGREES)

classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

TARGET_OBJECTS = ['mouse']

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def rad_to_deg(radians):
        return float(radians) * (180/math.pi)


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
    FOV_DEG_X = 110.0
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
    
    last_seen_pos = {
    "wrist_flex":0.0,
    "gripper":-62.0,
    "wrist_roll":3.2,
    }
    
    
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2, objects=TARGET_OBJECTS)

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
                last_seen_pos[arm.joint_names[1]] = rad_to_deg(shoulder_pan_angle)
                
                # Calculate shoulder lift triangle lengths
                sl_opposite = TRIG_MEASUREMENTS["base_to_tip"] - TRIG_MEASUREMENTS["ground_to_shoulder"]/2
                sl_adjacent = sp_hypotenuse
                sl_hypotenuse = math.sqrt((sl_opposite**2 + sl_adjacent**2))
                
                print(f"Conceptual shoulder lift triangle has lengths: opposite - {sl_opposite}, adjacent - {sl_adjacent}, hypotenuse - {sl_hypotenuse}")
                
                # Calculate elbow lift triangle
                el_c = sl_hypotenuse
                el_a = TRIG_MEASUREMENTS["lower_arm"]
                el_b = TRIG_MEASUREMENTS["forearm"]
                
                print(f"Conceptual elbow lift triangle has lengths: a - {el_a}, b - {el_b}, c - {el_c}")
                
                if el_a + el_b < el_c:
                    print("Outside of range")
                else:
                
                    thetaB = math.acos((el_a**2 + el_c**2 - el_b**2)/(2*el_a*el_c))
                    shoulder_lift_angle = math.pi - (thetaB + math.atan2(sl_opposite, sl_adjacent) - 0.22) # Subtract 12 degree motor error offset
                    last_seen_pos[arm.joint_names[2]] = rad_to_deg(shoulder_lift_angle)
                    
                    elbow_lift_angle = math.acos((el_a**2 + el_b**2 - el_c**2)/(2*el_b*el_a)) # Law of cosines to find theta(c)
                    last_seen_pos[arm.joint_names[3]] = rad_to_deg(elbow_lift_angle) * -1
                    
                    # Calculate wrist flex angle
                    # We can just add theta(a) from elbow lift and the angle from the shoulder lift calulation
                    
                    sl_part = math.atan2(sl_adjacent, sl_opposite)
                    thetaA = math.acos(((el_c**2 + el_b**2 - el_a**2)/(2*el_b*el_c)))
                    
                    wrist_flex_angle = sl_part + thetaA
                    #last_seen_pos[arm.joint_names[4]] == rad_to_deg(wrist_flex_angle) * -1
                
                
                    print(
                        f"Detected {class_name} (conf={conf:.2f}) -> "
                        f"target_base x={target['x']:.3f} y={target['y']:.3f} z={target['z']:.3f} "
                        f"(theta1={target['theta1_deg']:.1f} deg)\n"
                        f"{'*' * 10}\nLast Seen Pos Dict: {last_seen_pos}\n{'*' * 10}\n"
                        #f"=> Shoulder pan angle: {shoulder_pan_angle}\nShoulder lift angle: {shoulder_lift_angle}\nElbow lift angle: {elbow_lift_angle}\nWrist flex angle: {wrist_flex_angle}"
                    )
        
        
        
        cv2.imshow("Output",img)
        key = cv2.waitKey(1)
        
        #print(f"Detected key input: {key}")
        
        if key == 0xA or key == 0xD:
            print("Enter pressed, going to last seen object")
            
            print(f"Sending positions: {last_seen_pos}")
            arm.move_to_pose(last_seen_pos, duration=0)
            time.sleep(5)
            arm.move_to_pose(arm.rest_position_degrees, duration=0)
            
        if key == 0x71:
            print("Quitting...")
            
            arm.cleanup()
            break
    
