
import time
import mujoco
import numpy as np

def Rx(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def Ry(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
    
def Rz(thetadeg):
    thetarad = np.deg2rad(thetadeg)
    c = np.cos(thetarad)
    s = np.sin(thetarad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

# Displacement and rotation based on diagram at https://maegantucker.com/ECE4560/assets/assignment7/new-config.png
# Some of these values will need to be inverted based on testing
# Based on the assumption that x = red, y = green, z = blue
def get_gw1(theta1_deg):
    displacement = (0.0388353, 0.0, 0.0624) # translation from world frame to shoulder_pan motor
    rotation = Rx(180) @ Rz(180) @ Rz(theta1_deg) #Rotate 180 degrees in the z and x axes to account for the orientation of the motor in relation to the world frame, then rotate by theta1 around the z axis
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g12(theta2_deg):
    displacement = (-0.0303992, 0.0,  -0.0542) # translation from shoulder_pan motor to shoulder_lift motor
    rotation = Rz(-90) @ Ry(270) @ Rz(theta2_deg) # rotate from shoulder_pan frame to shoulder_lift frame, then rotate by theta2 around the z axis
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g23(theta3_deg):
    displacement = (-0.11257, -0.028, 0.0) # translation from shoulder_lift motor to elbow_flex motor
    rotation = Rz(90) @ Rz(theta3_deg) # rotate from shoulder_lift frame to elbow_flex frame, then rotate by theta3 around the y axis
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g34(theta4_deg):
    displacement = (-0.14, 0.0, 0.0) # translation from elbow_flex motor to wrist_flex motor - 2/24/26 changed from 0.1349 to .14 and the simulation looks more accurate
    rotation = Rz(-90) @ Rz(theta4_deg) # rotate from elbow_flex frame to wrist_flex frame, then rotate by theta4 around the y axis
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g45(theta5_deg):
    displacement = (0.0, -0.0611, 0.0) # translation from wrist_flex motor to wrist_roll motor (assumed to be zero since they are co-located)
    rotation = Rz(180) @ Rx(90) @ Rz(theta5_deg) # rotate from wrist_flex frame to wrist_roll frame, then rotate by theta5 around the z axis
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_g5t():
    displacement = (0.0, 0.0, -0.1034) # translation from wrist_roll motor to object frame
    rotation = Rx(90) # No rotation from wrist roll frame to object frame in new config
    pose = np.block([[rotation, np.array(displacement).reshape(3,1)], [0, 0, 0, 1]])
    return pose

def get_forward_kinematics(position_dict):
    gw1 = get_gw1(position_dict['shoulder_pan'])
    g12 = get_g12(position_dict['shoulder_lift'])
    g23 = get_g23(position_dict['elbow_flex'])
    g34 = get_g34(position_dict['wrist_flex'])
    g45 = get_g45(position_dict['wrist_roll'])
    g5t = get_g5t()
    gwt = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t
    position = gwt[0:3, 3]
    rotation = gwt[0:3, 0:3]
    return position, rotation
