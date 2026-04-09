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
        return calibration
    
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
        print(f"Creating Robot Interface on port: {port}")
        self.port = port
        self.motors = motors
        self.connected = False
        self.bus = None
        self.name = name
        
        self.joint_names = {
            1: "shoulder_pan",
            2: "shoulder_lift",
            3: "elbow_flex",
            4: "wrist_flex",
            5: "wrist_roll",
            6: "gripper",
        }
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

        # Attempt to instantiate a motors bus in a few ways:
        try:
            # common constructor pattern: MotorsBus(port=..., brand="feetech", model="sts3215", baudrate=...)
            # This may differ by lerobot version. Adjust if your installation uses another signature.
            try:
                self.bus = MotorsBus(port=self.port, motors=self.motors, brand="feetech", model="sts3215")
            except TypeError:
                # alternative: simpler signature
                print("Trying simpler signature")
                self.bus = FeetechMotorsBus(self.port, motors=self.motors, calibration=load_calibration(self.name))
                self.bus.connect()
            self.connected = True
            print(f"Connected to Motors bus on {self.port}.")
        except Exception as e:
            print("ERROR connecting to MotorsBus:", e)

    def cleanup(self):
        print("\r\n====================\r\nDisconnecting from arm\r\n====================")
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
            self.bus.sync_write("Goal_Position", desired_position, normalize=False)
        else:
            # Interpolate positions based on given duration
            start_time = time.time()
            starting_pose = self.bus.read("Present_Position", "wrist_flex", normalize=False)
            
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
