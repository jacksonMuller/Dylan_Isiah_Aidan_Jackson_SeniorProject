"""
engineering_showcase_demo.py - Revamped demo with modes for all three versions of control.
	- For the original keyboard demo, use '--mode motor'
	- For keyboard control using a conceptual point and inverse kinematics, use '--mode ik'
	- For camera control using calculated coordinates, use '--mode cv'
"""
import argparse
import sys
import time
import math
import tty
import termios
from contextlib import contextmanager

from utils.arm_interface import TRIG_MEASUREMENTS, RobotMotorInterface, DEFAULT_MOTORS_DEGREES, rad_to_deg, DEFAULT_JOINT_NAMES, DEFAULT_MOTORS
import testCoordinate

VERBOSE = 0

def usage(mode):
    if mode == "ik":
        print(
        """
        Robotic Arm Keyboard Controls:
        \r\n a / d : X coordinate control
        \r\n w / s : Y coordinate control
        \r\n g : reach down and grab
        \r\n r : release gripper
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
    elif mode == 'cv':
        print(
        """
        Robotic Arm CV Control:
        \r\n Place mouse in view of the camera and press enter
        """
        )
    else:
        print(f"Unknown mode: {mode}")
    
@contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    try:
        tty.setraw(file.fileno())
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)
    
def ik_keyboard_control():
    arm = RobotMotorInterface(motors=DEFAULT_MOTORS_DEGREES)
    
    try:
        import select
    except Exception:
        print("select module missing - keyboard input may not work correctly")
        select = None
        
    point = [0, 0.1, (TRIG_MEASUREMENTS["base_to_tip"] - TRIG_MEASUREMENTS["ground_to_shoulder"]/2)]
    
    arm.move_to_pose(arm.get_target_angles(point[0], point[1]), duration=0)
    
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
                    
                if VERBOSE: print(f"Current Point: {point}")
                
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
                        if VERBOSE: print("\r\n +Y")
                        point[1] += 0.01
                    case 's':
                        if VERBOSE: print("\r\n -Y")
                        point[1] -= 0.01
                    case 'a':
                        if VERBOSE: print("\r\n -X")
                        point[0] -= 0.01
                    case 'd':
                        if VERBOSE: print("\r\n +X")
                        point[0] += 0.01
                    case '\x20':
                        point[2] += 0.01
                    case 'c':
                        point[2] -= 0.01
                    case 'g':
                        arm.claw_machine_grab(point[0], point[1], point[2])
                    case 'r':
                        arm.release_gripper()
                    case _:
                        print(f"No command mapped to: {ch}")
                        
                point[2] = max(point[2], (TRIG_MEASUREMENTS["base_to_tip"] - TRIG_MEASUREMENTS["ground_to_shoulder"]))
                        
                arm.move_to_pose(arm.get_target_angles(point[0], point[1], point[2]), duration=0)
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
        if VERBOSE: print("select module missing — keyboard input may not work correctly.")
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
                    
                if VERBOSE: print("\r\nCurrent key: ", ch)

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
                        if VERBOSE: print("\r\nPan right")
                        interface.motor_control(1, -30)
                    case 'd':
                        if VERBOSE: print("\r\nPan right")
                        interface.motor_control(1, 30)
                    case 'w':
                        if VERBOSE: print("\r\nShoulder up")
                        interface.motor_control(2, 80)
                    case 's':
                        if VERBOSE: print("\r\nShoulder down")
                        interface.motor_control(2, -80)
                    case 'y':
                        if VERBOSE: print("\r\nElbow up")
                        interface.motor_control(3, -80)
                    case 'h':
                        if VERBOSE: print("\r\nElbow down")
                        interface.motor_control(3, 80)
                    case 'j':
                        if VERBOSE: print("\r\nWrist right")
                        interface.motor_control(5, 60)
                    case 'l':
                        if VERBOSE: print("\r\nWrist left")
                        interface.motor_control(5, -60)
                    case 'i':
                        if VERBOSE: print("\r\nWrist up")
                        interface.motor_control(4, -60)
                    case 'k':
                        if VERBOSE: print("\r\nWrist down")
                        interface.motor_control(4, 60)
                    case 'q':
                        if VERBOSE: print("\r\nClosing hand")
                        interface.motor_control(6, 15)
                    case 'e':
                        if VERBOSE: print("\r\nOpening hand")
                        interface.motor_control(6, -15)
                    case 'r':
                        if VERBOSE: print("\r\nResting arm")
                        interface.rest_arm()
                    case '1':
                        if VERBOSE: print("\r\nWave")
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
    elif mode == "cv":
        testCoordinate.main()
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ik", help="Operation mode - ik for inverse kinematic mode, motor for individual control")
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbose mode - 1 for debug messages, 0 for quiet mode")
    args = parser.parse_args()
    VERBOSE = args.verbose
    
    main(mode=args.mode)
