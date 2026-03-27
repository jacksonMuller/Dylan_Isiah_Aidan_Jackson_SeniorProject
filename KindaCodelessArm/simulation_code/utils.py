import math
import mujoco
import time 

'''
File that handles low-level communication with MuJoCo. 
Everything else calls functions from here
'''

def degrees_to_radians(degrees_dict: dict[str, int]) -> dict[str, float]:
    '''
    Function that converts joint angles in degrees to radians
    Input: degrees_dict: dict[str, float]
    Output: result: dict[str, float]
    '''
    result: dict[str, float] = {}

    for joint_name, angle_in_degrees in degrees_dict.items():
        result[joint_name] = angle_in_degrees * (math.pi / 180.0)

    return result

def radians_to_degrees(radians_dict: dict[str, float]) -> dict[str, float]: 
    '''
    Function that converts joint angles in radians to degrees
    Input: radians_dict: dict[str, float]
    Output: result: dict[str, float]
    '''
    result: dict[str, float] = {}

    for joint_name, angle_in_radians in radians_dict.items(): 
        result[joint_name] = angle_in_radians * (180.0 / math.pi)

    return result

def get_current_joint_angles(m, d): 
    '''
    Returns where all the joints are right now, and returns them as a dictionary in degrees
    Input: m: mjModel -> Instance of model object (The blueprint of the robot)
    Input: d: mjData -> Runtime simulation state (The live state of the simulation right now)
    Output: result: dict[str, float] (in degrees)
    '''
    # Create array that keeps track of the 6 joint angles in radians
    raw = d.qpos 

    angles_in_rad: dict[str, float] = {
        'shoulder_pan': raw[0],
        'shoulder_lift': raw[1],
        'elbow_flex': raw[2],
        'wrist_flex': raw[3],
        'wrist_roll': raw[4],
        'gripper': raw[5]
    }

    # Return dict in degrees
    return radians_to_degrees(angles_in_rad)

def send_joint_command(m, d, joint_angles_deg: dict[str, float]) -> None:
    '''
    Takes desired joint angles (in degrees), converts to radians, and writes them to d.ctrl (so motors begin moving)
    Input: m: mjModel -> Instance of model object (The blueprint of the robot)
    Input: d: mjData -> Runtime simulation state (The live state of the simulation right now)
    Input: joint_angles_deg --> Dict of current joint angles in degrees
    Output: None
    '''
    angles_in_rad: dict[str, float] = degrees_to_radians(joint_angles_deg)

    d.ctrl[0] = angles_in_rad['shoulder_pan']
    d.ctrl[1] = angles_in_rad['shoulder_lift']
    d.ctrl[2] = angles_in_rad['elbow_flex']
    d.ctrl[3] = angles_in_rad['wrist_flex']
    d.ctrl[4] = angles_in_rad['wrist_roll']
    d.ctrl[5] = angles_in_rad['gripper']

def move_to_pose(m, d, viewer, target_angles_deg: dict[str, float], duration_sec: float) -> None: 
    '''
    Function interpolates from the current joint angles to target angles over duration_sec seconds. 
    Helps prevent the arm from snapping quickly to a new position
    Input: m: mjModel -> Instance of model object (The blueprint of the robot)
    Input: d: mjData -> Runtime simulation state (The live state of the simulation right now)
    Input: viewer: ## FILL IN ## 
    Input: target_angles_deg --> Dict of target angles in degrees 
    Input: duration_sec --> Duration for the arm to move to target_angles_deg
    Output: None 
    '''
    # Keep track of when we start 
    start_time = time.time()

    # Get the current angles of the arm
    start_angles = get_current_joint_angles(m, d)

    # Loop until we reach achieved position
    while True: 
        # Keep track of how long we've been in the loop
        elapsed_time = time.time() - start_time 

        # Check to see if we've hit the duration_secs
        if elapsed_time >= duration_sec:
            break 

        alpha = elapsed_time / duration_sec
        alpha = max(0.0, min(alpha, 1.0))

        # Blend between start and target for each joint
        interpolated: dict[str, float] = {}

        # Loop through each joint and calculate the angle to move motor to target
        for joint in target_angles_deg:
            start_val = start_angles[joint]
            target_val = target_angles_deg[joint]
            interpolated[joint] = start_val * (1 - alpha) + target_val * alpha
    
        send_joint_command(m, d, interpolated)
        mujoco.mj_step(m, d)
        viewer.sync()

def hold_position(m, d, viewer, duration_sec: float) -> None: 
    '''
    Function allows the robot to hold position over duration_sec
    Input: m: mjModel -> Instance of model object (The blueprint of the robot)
    Input: d: mjData -> Runtime simulation state (The live state of the simulation right now)
    Input: viewer: ## FILL IN ## 
    Input: duration_sec --> How long arm should hold the position 
    Output: None 
    '''
    current_angles = get_current_joint_angles(m, d)

    start_time = time.time()

    # Loop until we hit duration_sec
    while True: 
        # Keep track of how long we've been in the loop
        elapsed_time = time.time() - start_time
        if elapsed_time >= duration_sec:
            break 

        # We send position to send_joint_command, so the motors stay put
        send_joint_command(m, d, current_angles)
        mujoco.mj_step(m, d)
        viewer.sync()

def get_gripper_position(m ,d) -> list[float]:
    '''
    Returns the [x, y, z] position of the grippers location
    Input: m: mjModel -> Instance of model object (The blueprint of the robot)
    Input: d: mjData -> Runtime simulation state (The live state of the simulation right now)
    Output: list[float] --> [x, y, z] position of the gripper at the current state (in meters)
    '''
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    # [x, y, z] in meters
    position = d.site_xpos[site_id]
    return position

        




