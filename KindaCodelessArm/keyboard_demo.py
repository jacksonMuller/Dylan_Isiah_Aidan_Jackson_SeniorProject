#!/usr/bin/env python3
"""
so101_keyboard_teleop.py

Run on a Raspberry Pi connected to a single SO-101 arm (Feetech servos).
Teleoperates joint angles with keyboard keys.

Usage:
    python3 so101_keyboard_teleop.py --port /dev/ttyACM0

Notes:
 - Requires lerobot installed (or components from the repo).
 - You may need root or to chmod the serial port: sudo chmod 666 /dev/ttyACM0
 - If lerobot APIs differ slightly on your version, see the fallback functions below.
"""
import argparse
import sys
import time
import threading
import math

# keyboard input helpers (unix)
import tty
import termios
from contextlib import contextmanager
from lerobot.motors import Motor, MotorNormMode

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
    
motors={
                "joint_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_4": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_5": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_6": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
            }

# helper to read single char (unix)
@contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    try:
        tty.setraw(file.fileno())
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

def read_char(timeout=None):
    """Read one char from stdin, non-blocking with optional timeout (seconds)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if timeout is None:
            ch = sys.stdin.read(1)
            return ch
        else:
            end = time.time() + timeout
            while time.time() < end:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    return sys.stdin.read(1)
            return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# A lightweight wrapper to send positions; tries a few likely APIs.
class RobotMotorInterface:
    def __init__(self, port="/dev/ttyACM0", motors=None):
        """
        port: port of motor control board.  Use lerobot calibration script to find correct port.  For more instructions see the huggingface website.
        motors: mapping of motor names to motor objects
        """
        self.port = port
        self.motors = motors
        self.connected = False
        self.bus = None
        
        self.joint_names = {
            1: "joint_1",
            2: "joint_2",
            3: "joint_3",
            4: "joint_4",
            5: "joint_5",
            6: "joint_6",
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
                self.bus = FeetechMotorsBus(self.port, motors=self.motors)
                self.bus.connect()
            self.connected = True
            print(f"Connected to Motors bus on {self.port}.")
        except Exception as e:
            print("ERROR connecting to MotorsBus:", e)
            
    def cleanup(self):
        print("\r\n====================\r\nDisconnecting from arm\r\n====================")
        self.rest_arm()
        self.bus.disconnect()
        
    """

    Motor Control Functions

    """
    

    def motor_control(self, motor, offset):
        """
            Generic motor control function
            motor: number corresponding to a motor in joint_names
            offset: change in position you want to give the motor
        """
        motor_pos = self.bus.read("Present_Position", self.joint_names[motor], normalize=False)
        print("\r\nCurrent position of", self.joint_names[motor], motor_pos)
        self.bus.write("Goal_Position", self.joint_names[motor], motor_pos+offset, normalize=False)
        
    def pan_arm(self, offset):
        position = self.bus.read("Present_Position", "joint_1", normalize=False)
        print("\r\nCurrent position: ", position)
        self.bus.write("Goal_Position", "joint_1", position+offset, normalize=False)
        
    def extend_shoulder(self, offset):
        j2_position = self.bus.read("Present_Position", "joint_2", normalize=False)
        print("\r\nCurrent j2 position: ", j2_position)
        self.bus.write("Goal_Position", "joint_2", j2_position+offset, normalize=False)
        
    def extend_elbow(self, offset):
        j3_position = self.bus.read("Present_Position", "joint_3", normalize=False)
        print("\r\nCurrent j3 position: ", j3_position)
        self.bus.write("Goal_Position", "joint_3", j3_position+offset, normalize=False)
        
    def twist_wrist(self, offset):
        j5_position = self.bus.read("Present_Position", "joint_5", normalize=False)
        print("\r\nCurrent position: ", j5_position)
        self.bus.write("Goal_Position", "joint_5", j5_position+offset, normalize=False)
        
    def flex_wrist(self, offset):
        j4_position = self.bus.read("Present_Position", "joint_4", normalize=False)
        print("\r\nCurrent position: ", j4_position)
        self.bus.write("Goal_Position", "joint_4", j4_position+offset, normalize=False)
        
    def hand_control(self, offset):
        j6_position = self.bus.read("Present_Position", "joint_6", normalize=False)
        print("\r\nCurrent position: ", j6_position)
        self.bus.write("Goal_Position", "joint_6", j6_position+offset, normalize=False)
        
    def wave_hand(self):
        """
            Goes to hardcoded wave position and then moves wrist back and forth to "wave"
            Positions for raised hand:
                Joint_1: 2175
                Joint_2: 1354
                Joint_3: 2014
                Joint_4: Moving joint
                Joint_5: 3786
                Joint_6: 2395
        """
        print("\r\nStarting Wave")
        self.bus.sync_write("Goal_Position", self.wave_starting_position, normalize=False)
        time.sleep(1)
        j4_position = self.bus.read("Present_Position", "joint_4", normalize=False)
        for i in range(3):
            print("\r\nWaving Hand")
            self.bus.write("Goal_Position", "joint_4", j4_position+200, normalize=False)
            time.sleep(1)
            self.bus.write("Goal_Position", "joint_4", j4_position-200, normalize=False)
            time.sleep(1)
        self.bus.write("Goal_Position", "joint_4", j4_position, normalize=False)
        
    def rest_arm(self):
        """
        Goes to hardcoded resting position
        """
        print("\r\nGoing to rest")
        self.bus.sync_write("Goal_Position", self.rest_position, normalize=False)
        
            
        

# human-friendly help
def print_help():
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None, help="Serial port for MotorsBus (e.g. /dev/ttyACM0)")
    args = parser.parse_args()

    print("LeRobot SO-101 keyboard teleop (keyboard control).")
    print_help()

    interface = RobotMotorInterface(port="/dev/ttyACM0", motors=motors)
    print("Press keys to move joints. Ctrl-C or ESC to quit.")

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
        interface.cleanup()
        print("Exiting teleop.")

if __name__ == "__main__":
    main()
