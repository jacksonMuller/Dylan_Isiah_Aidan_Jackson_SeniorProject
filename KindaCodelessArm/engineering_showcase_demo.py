# Updated keyboard demo with inverse kinematics
import argparse
import sys
import time
import math
import tty
import termios
from contextlib import contextmanager

from utils.arm_interface import TRIG_MEASUREMENTS, RobotMotorInterface, DEFAULT_MOTORS_DEGREES, rad_to_deg, DEFAULT_JOINT_NAMES, DEFAULT_MOTORS

def usage(mode):
    if mode == "ik":
        print(
        """
        Robotic Arm Keyboard Controls:
        \r\n a / d : X coordinate control
        \r\n w / s : Y coordinate control
        \r\n Ctrl-C / esc : exit
        """
        )
    elif mode == "motor":
        print(
        """SO-101 keyboard teleop keys:
        \r\n  a / d : Pan left/right
        \r\n  w / s : Shoulder up/down
        \r\n  y / h : Elbow up/down
        \r\n  i / k : Wrist up/down
        \r\n  j / l : Wrist twist
        \r\n  q / e : gripper
        \r\n  r : return to resting position
        \r\n  1 : wave
        \r\n  Esc or Ctrl-C : exit\r\n"""
        )
    
@contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    try:
        tty.setraw(file.fileno())
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

def get_target_angles(pointX, pointY):
    """
    Get motor angles in degrees to place the hand over the specified point
    
    Arguments:
        - pointX: X (left-right from the perspective of the arm base) coordinate of point in m with the origin at the point of rotation in the shoulder_pan motor
        - pointY: Y (forward-backward from the perspective of the arm base) coordinate of point in m with the origin at the point of rotation in the shoulder_pan motor
        
    Returns:
        - pos: dictionary of angles for each motor in a format that can be passed directly to move_to_pose on the arm interface
    """
    
    #Initialize dict with values that aren't going to change
    pos = {
    "wrist_flex": 0.0,
    "gripper": -62.0,
    "wrist_roll": 3.2,
    }
    
    # Shoulder_pan - 2d projected triangle on the surface of the table
    sp_adjacent = pointY
    sp_opposite = pointX
    sp_hypotenuse = math.sqrt((sp_adjacent**2 + sp_opposite**2))
                
    shoulder_pan_angle = math.atan2(sp_opposite, sp_adjacent)
    pos[DEFAULT_JOINT_NAMES[1]] = rad_to_deg(shoulder_pan_angle)
    
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
    
    if el_a + el_b < el_c: # Catch if we're trying to reach past where the two segments of the arm can reach to prevent domain err on the arccos call
        print("Outside of range")
    else:
    
        thetaB = math.acos((el_a**2 + el_c**2 - el_b**2)/(2*el_a*el_c))
        shoulder_lift_angle = math.pi - (thetaB + math.atan2(sl_opposite, sl_adjacent) - 0.22) # Subtract 12 degree motor error offset to calibrate to level, then subtract calculated value from 180
        pos[DEFAULT_JOINT_NAMES[2]] = rad_to_deg(shoulder_lift_angle)
        
        elbow_lift_angle = math.acos((el_a**2 + el_b**2 - el_c**2)/(2*el_b*el_a)) # Law of cosines to find theta(c)
        pos[DEFAULT_JOINT_NAMES[3]] = rad_to_deg(elbow_lift_angle) * -1
    
    return pos
    
def ik_keyboard_control():
    arm = RobotMotorInterface(motors=DEFAULT_MOTORS_DEGREES)
    
    try:
        import select
    except Exception:
        print("select module missing - keyboard input may not work correctly")
        select = None
        
    point = [0, 0.1]
    
    arm.move_to_pose(get_target_angles(point[0], point[1]), duration=0)
    
    try:
        with raw_mode(sys.stdin):
            while True:
                if select:
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r:
                        continue
                    ch = sys.stdin.read(1)
                else:
                    ch = sys.stdin.read(1)
                    
                print(f"Current Point: {point}")
                
                if not ch:
                    continue
                if ch == '\x1b':
                    print("ESC pressed - exiting")
                    break
                if ch == '\x03':
                    print("Ctrl-C pressed - exiting")
                    break
                    
                match ch:
                    case 'w':
                        print("\r\n +Y")
                        point[1] += 0.01
                    case 's':
                        print("\r\n -Y")
                        point[1] -= 0.01
                    case 'a':
                        print("\r\n -X")
                        point[0] -= 0.01
                    case 'd':
                        print("\r\n +X")
                        point[0] += 0.01
                    case _:
                        print(f"No command mapped to: {ch}")
                        
                arm.move_to_pose(get_target_angles(point[0], point[1]), duration=0)
    except KeyboardInterrupt:
        print("Interrupted - exiting")
    finally:
        arm.move_to_pose(arm.rest_position_degrees)
        time.sleep(2)
        arm.cleanup()

def motor_keyboard_control():
    interface = RobotMotorInterface(motors=DEFAULT_MOTORS)
    try:
        import select
    except Exception:
        print("select module missing — keyboard input may not work correctly.")
        select = None

    try:
        with raw_mode(sys.stdin):
            while True:
                if select:
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r:
                        continue
                    ch = sys.stdin.read(1)
                else:
                    ch = sys.stdin.read(1)
                    
                print("\r\nCurrent key: ", ch)

                if not ch:
                    continue
                # handle exit
                if ch == '\x1b':  # ESC
                    print("ESC pressed — exiting")
                    break
                if ch == '\x03':  # Ctrl-C
                    print("Ctrl-C — exiting")
                    break
                    
                #The commands below this are the only ones that work.  Use the update position function (or similar code) to change motor values.  Examples in the comments of the motor_bus header explain that code
                #TODO: Find a way to interpolate multiple keypresses at the same time.
                
                #Keyboard Control
                match ch:
                    case 'a':
                        print("\r\nPan right")
                        interface.motor_control(1, -30)
                    case 'd':
                        print("\r\nPan right")
                        interface.motor_control(1, 30)
                    case 'w':
                        print("\r\nShoulder up")
                        interface.motor_control(2, 80)
                    case 's':
                        print("\r\nShoulder down")
                        interface.motor_control(2, -80)
                    case 'y':
                        print("\r\nElbow up")
                        interface.motor_control(3, -80)
                    case 'h':
                        print("\r\nElbow down")
                        interface.motor_control(3, 80)
                    case 'j':
                        print("\r\nWrist right")
                        interface.motor_control(5, 60)
                    case 'l':
                        print("\r\nWrist left")
                        interface.motor_control(5, -60)
                    case 'i':
                        print("\r\nWrist up")
                        interface.motor_control(4, -60)
                    case 'k':
                        print("\r\nWrist down")
                        interface.motor_control(4, 60)
                    case 'q':
                        print("\r\nClosing hand")
                        interface.motor_control(6, 15)
                    case 'e':
                        print("\r\nOpening hand")
                        interface.motor_control(6, -15)
                    case 'r':
                        print("\r\nResting arm")
                        interface.rest_arm()
                    case '1':
                        print("\r\nWave")
                        interface.wave_hand()
                    case _:
                        print("No command mapped to: ", ch)
                        
    except KeyboardInterrupt:
        print("Interrupted — exiting")
    finally:
        interface.move_to_pose(interface.rest_position_degrees)
        time.sleep(2)
        interface.cleanup()

def main(mode):
    usage(mode)
    if mode == "ik":
        ik_keyboard_control()
    elif mode == "motor":
        motor_keyboard_control()
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ik", help="Operation mode - ik for inverse kinematic mode, motor for individual control")
    args = parser.parse_args()
    
    main(mode=args.mode)
