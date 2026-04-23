import argparse
import sys
import time
import threading
import math
import draccus
from pathlib import Path

# keyboard input helpers (unix)
import tty
import termios
from contextlib import contextmanager
from lerobot.motors import Motor, MotorNormMode, MotorCalibration

# Try to import lerobot motor bus classes (multiple possible import paths).
MotorsBus = None
FeetechMotorsBus = None
_has_lerobot = False

try:
    # primary guess: recent lerobot layout
    from lerobot.common.motors.motors_bus import MotorsBus
    _has_lerobot = True
except Exception:
    try:
        from lerobot.motors.feetech import FeetechMotorsBus
        MotorsBus = FeetechMotorsBus
        _has_lerobot = True
    except Exception:
        _has_lerobot = False

if not _has_lerobot:
    print("Warning: could not import lerobot MotorsBus classes. "
          "The script will still run in 'dry' mode, but cannot send commands to the arm.\n"
          "Install Lerobot (and the feetech extras) with:\n"
          "  pip install -e '.[feetech]'\n"
          "or follow the repo instructions: https://github.com/huggingface/lerobot\n")
    MotorsBus = None  # keep name defined
    
def load_calibration(robot_name="KindaCodeless"):
    curr_path = Path(__file__).resolve().parent
    fpath = curr_path / f"{robot_name}Config.json"
    with open(fpath) as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)
        #print(f"Loaded calibration: {calibration}")
        return calibration
        
def rad_to_deg(radians):
        return float(radians) * (180/math.pi)
    
DEFAULT_MOTORS={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
            }
            
DEFAULT_MOTORS_DEGREES={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.DEGREES),
            }
            
DEFAULT_JOINT_NAMES = {
            1: "shoulder_pan",
            2: "shoulder_lift",
            3: "elbow_flex",
            4: "wrist_flex",
            5: "wrist_roll",
            6: "gripper",
        }

TRIG_MEASUREMENTS={ # Important measurements for making conceptual triangles with the robot
                "ground_to_shoulder": 0.125, # Height from the ground to the point of rotation on shoulder_lift motor
                "lower_arm": 0.125, # Length of lower arm from shoulder_lift to elbow_flex
                "forearm": 0.145, # Length of forearm from elbow_flex to wrist_flex
                "base_to_tip": 0.17, # Length from wrist_flex to tip of grabber
}

class RobotMotorInterface:
    """
    RobotMotorInterface: Main interface helper class that contains easy abstractions
                         for connecting to and controlling the arm.
    """
    def __init__(self, port="/dev/ttyACM0", motors=DEFAULT_MOTORS, name="KindaCodeless"):
        """
        port: port of motor control board.  Use lerobot calibration script to find correct port.  For more instructions see the huggingface website.
        motors: mapping of motor names to motor objects
        """
        #print(f"Creating Robot Interface on port: {port}")
        self.port = port
        self.motors = motors
        self.connected = False
        self.bus = None
        self.name = name
        
        self.joint_names = DEFAULT_JOINT_NAMES
        self.wave_starting_position = {
            self.joint_names[1]: 2175,
            self.joint_names[2]: 1354,
            self.joint_names[3]: 2014,
            self.joint_names[4]: 1545,
            self.joint_names[5]: 3786,
            self.joint_names[6]: 2395,
        }
        self.rest_position = {
            self.joint_names[1]: 2009,
            self.joint_names[2]: 888,
            self.joint_names[3]: 2990,
            self.joint_names[4]: 2792,
            self.joint_names[5]: 2984,
            self.joint_names[6]: 2045,
        }
        self.rest_position_degrees = {
            self.joint_names[1]: -8.0, 
            self.joint_names[2]: -0.3956043956043956, 
            self.joint_names[3]: -0.5714285714285714, 
            self.joint_names[4]: 0.21978021978021978, 
            self.joint_names[5]: -1.6263736263736264, 
            self.joint_names[6]: -61.27472527472528
        }

        try:
            try:
                self.bus = MotorsBus(port=self.port, motors=self.motors, brand="feetech", model="sts3215")
            except TypeError:
                # print("Trying simpler signature")
                self.bus = FeetechMotorsBus(self.port, motors=self.motors, calibration=load_calibration(self.name))
                self.bus.connect()
            self.connected = True
            #print(f"Connected to Motors bus on {self.port}.")
        except Exception as e:
            print("ERROR connecting to MotorsBus:", e)

    def cleanup(self):
        print("\r\n======================\r\nDisconnecting from arm\r\n======================")
        self.bus.disconnect()

    def motor_control(self, motor, offset):
        """
            Ultra-simple generic motor control function with no smoothing.
            Controls one motor at a time and moves it by a given offset

            Arguments:
                - motor: number corresponding to a motor in joint_names
                - offset: change in position you want to give the motor
        """
        motor_pos = self.bus.read("Present_Position", self.joint_names[motor], normalize=False)
        print("\r\nCurrent position of", self.joint_names[motor], motor_pos)
        self.bus.write("Goal_Position", self.joint_names[motor], motor_pos+offset, normalize=False)

    def wave_hand(self):
        """
            Goes to hardcoded wave position and then moves wrist back and forth to "wave"
            Positions for raised hand:
                shoulder_pan: 2175
                shoulder_lift: 1354
                elbow_flex: 2014
                wrist_flex: Moving joint
                wrist_roll: 3786
                gripper: 2395
        """
        print("\r\nStarting Wave")
        self.bus.sync_write("Goal_Position", self.wave_starting_position, normalize=False)
        time.sleep(1)
        j4_position = self.bus.read("Present_Position", "wrist_flex", normalize=False)
        for _ in range(3):
            print("\r\nWaving Hand")
            self.bus.write("Goal_Position", "wrist_flex", j4_position+200, normalize=False)
            time.sleep(1)
            self.bus.write("Goal_Position", "wrist_flex", j4_position-200, normalize=False)
            time.sleep(1)
        self.bus.write("Goal_Position", "wrist_flex", j4_position, normalize=False)
        
    def rest_arm(self):
        """
        Goes to hardcoded resting position
        """
        print("\r\nGoing to rest")
        self.bus.sync_write("Goal_Position", self.rest_position, normalize=False)
        
    def close_gripper(self, load_threshold=50, step=2.0, step_delay=0.05):
        current_pos = self.bus.read("Present_Position", "gripper")
        
        while current_pos > -62.0:
            try:
                load = self.bus.read("Present_Load", "gripper")
            except Exception as e:
                print(f"Load read error: {e}")
        
            print(f"\r\nGripper pos={current_pos:.1f}, load={load}")
            
            if abs(load) >= load_threshold:
                print("Object gripped")
                break
                
            current_pos -= step
            try:
                self.bus.write("Goal_Position", "gripper", max(current_pos, -62.0))
            except Exception as e:
                print(f"Write error (ignoring): {e}")
                break
            time.sleep(step_delay)
            
    def release_gripper(self):
        try:
            self.bus.write("Goal_Position", "gripper", 0.0)
        except Exception as e:
            print(f"Gripper release write error: {e}")
        time.sleep(1)
        try:
            self.bus.write("Goal_Position", "gripper", -62.0)
        except Exception as e:
            print(f"Gripper re-close write error: {e}")
        time.sleep(1)
        
    def claw_machine_grab(self, pointX, pointY, pointZ):
        """
        Opens gripper, reaches down, and tries to pick up something directly below the gripper
        """
        
        # Open gripper
        starting_pose = self.get_target_angles(pointX, pointY, pointZ)
        starting_pose["gripper"] = 0
        self.move_to_pose(starting_pose)
        time.sleep(1)
        
        # Reach down
        target_pose = self.get_target_angles(pointX, pointY, TRIG_MEASUREMENTS["base_to_tip"] - TRIG_MEASUREMENTS["ground_to_shoulder"])
        self.move_to_pose(target_pose)
        time.sleep(1)
        
        # Close gripper
        self.close_gripper()
        time.sleep(1)
        
        # Pick up
        closed_gripper_pos = self.bus.read("Present_Position", "gripper")
        starting_pose["gripper"] = closed_gripper_pos
        self.move_to_pose(starting_pose)
        time.sleep(1)
        
    def get_target_angles(self, pointX, pointY, pointZ=(TRIG_MEASUREMENTS["base_to_tip"] - TRIG_MEASUREMENTS["ground_to_shoulder"]/2)):
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
        pos[self.joint_names[1]] = rad_to_deg(shoulder_pan_angle)
        
        # Calculate shoulder lift triangle lengths
        sl_opposite = pointZ
        sl_adjacent = sp_hypotenuse
        sl_hypotenuse = math.sqrt((sl_opposite**2 + sl_adjacent**2))
        
        # Calculate elbow lift triangle
        el_c = sl_hypotenuse
        el_a = TRIG_MEASUREMENTS["lower_arm"]
        el_b = TRIG_MEASUREMENTS["forearm"]
        
        # Uncomment for debug messages about triangle side lengths
        #print(f"Conceptual shoulder lift triangle has lengths: opposite - {sl_opposite}, adjacent - {sl_adjacent}, hypotenuse - {sl_hypotenuse}\r\nConceptual elbow lift triangle has lengths: a - {el_a}, b - {el_b}, c - {el_c}")
        
        if el_a + el_b < el_c: # Catch if we're trying to reach past where the two segments of the arm can reach to prevent domain err on the arccos call
            print("Outside of range")
        else:
        
            thetaB = math.acos((el_a**2 + el_c**2 - el_b**2)/(2*el_a*el_c))
            shoulder_lift_angle = math.pi - (thetaB + math.atan2(sl_opposite, sl_adjacent) - 0.22) # Subtract 12 degree motor error offset to calibrate to level, then subtract calculated value from 180
            pos[self.joint_names[2]] = rad_to_deg(shoulder_lift_angle)
            
            elbow_lift_angle = math.acos((el_a**2 + el_b**2 - el_c**2)/(2*el_b*el_a)) # Law of cosines to find theta(c)
            pos[self.joint_names[3]] = rad_to_deg(elbow_lift_angle) * -1
            
            # Calculate wrist flex angle
            # We can just add theta(a) from elbow lift and the angle from the shoulder lift calulation
            
            sl_part = math.atan2(sl_adjacent, sl_opposite)
            thetaA = math.acos(((el_c**2 + el_b**2 - el_a**2)/(2*el_b*el_c)))
            
            wrist_flex_angle = sl_part + thetaA
            pos[self.joint_names[4]] = max(min(23.0, ((rad_to_deg(wrist_flex_angle) * -1) + 90)), -180.0)
            #print(max(min(23.0, ((rad_to_deg(wrist_flex_angle) * -1) + 90)), -180.0))
        
        return pos

    def move_to_pose(self, desired_position, duration=0):
        """
        Generic position input that should smooth the transition based on duration argument
        Arguments:
            - desired_position: dictionary with the target position, 
              see self.starting_position or self.rest_position for examples
            - duration: duration of movement, leave at 0 to just go immediately
        """

        if duration == 0:
            # Just go as fast as possible
            self.bus.sync_write("Goal_Position", desired_position, normalize=True)
        else:
            # Interpolate positions based on given duration
            start_time = time.time()
            starting_pose = self.bus.sync_read("Present_Position", normalize=True)
            
            while True:
                t = time.time() - start_time
                if t > duration:
                    break

                # Interpolation factor [0,1] (make sure it doesn't exceed 1)
                alpha = min(t / duration, 1)

                # Interpolate each joint
                position_dict = {}
                for joint in desired_position:
                    p0 = starting_pose[joint]
                    pf = desired_position[joint]
                    position_dict[joint] = (1 - alpha) * p0 + alpha * pf

                # Send command
                self.bus.sync_write("Goal_Position", position_dict, normalize=False)
