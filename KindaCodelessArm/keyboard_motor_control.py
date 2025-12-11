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
    def __init__(self, port=None, baudrate=1000000, motor_ids=None, dry=False):
        """
        motor_ids: list of motor ids (1..6) in the order expected by your arm.
        dry: if True, don't attempt to talk to hardware.
        """
        self.dry = dry or (MotorsBus is None)
        self.port = "/dev/ttyACM0"
        self.baudrate = baudrate
        self.motor_ids = motor_ids or [1,2,3,4,5,6]
        self.connected = False
        self.bus = None

        if self.dry:
            print("[dry mode] not connecting to hardware.")
            return

        # Attempt to instantiate a motors bus in a few ways:
        try:
            # common constructor pattern: MotorsBus(port=..., brand="feetech", model="sts3215", baudrate=...)
            # This may differ by lerobot version. Adjust if your installation uses another signature.
            try:
                self.bus = MotorsBus(port=self.port, motors=motors, brand="feetech", model="sts3215")
            except TypeError:
                # alternative: simpler signature
                print("Trying simpler signature")
                self.bus = FeetechMotorsBus(self.port, motors=motors)
                self.bus.connect()
            self.connected = True
            print(f"Connected to Motors bus on {self.port}.")
        except Exception as e:
            print("ERROR connecting to MotorsBus:", e)
            print("Proceeding in dry mode.")
            self.dry = True
            
    def cleanup(self):
        print("\r\n====================\r\nDisconnecting from arm\r\n====================")
        self.bus.disconnect()
        
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

# keyboard mapping (change as you like)
DEFAULT_KEYMAP = {
    # (key_positive, key_negative) : joint_index (0-based)
    ('q', 'u'): 0,   # base pan +/-
    ('w', 'i'): 1,   # shoulder lift
    ('e', 'o'): 2,   # elbow flex
    ('r', 'p'): 3,   # wrist flex
    ('t', 'l'): 4,   # wrist roll
    ('g', 'h'): 5,   # gripper open/close
}

# human-friendly help
def print_help():
    print("SO-101 keyboard teleop keys:")
    print("  a / d : Pan left/right")
    print("  w / s : Shoulder up/down")
    print("  y / h : Elbow up/down")
    print("  i / k : Wrist up/down")
    print("  j / l : Wrist twist")
    print("  q / e : gripper")
    print("  Esc or Ctrl-C : exit")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None, help="Serial port for MotorsBus (e.g. /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=1000000, help="Baudrate for motors bus")
    parser.add_argument("--dry", action="store_true", help="Don't send to hardware (dry run)")
    parser.add_argument("--step", type=float, default=3.0, help="Initial step size (degrees or percent, depending on your setup)")
    args = parser.parse_args()

    print("LeRobot SO-101 keyboard teleop (keyboard control).")
    print_help()

    motor_ids = [1,2,3,4,5,6]  # default—adjust if your motors use different id order
    interface = RobotMotorInterface(port=args.port, baudrate=args.baud, motor_ids=motor_ids, dry=args.dry)
    joints = [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]  # neutral-ish start; calibrate to your robot's zero if needed
    step = float(args.step)
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
                    
                    #The two commands below this are the only ones that work.  Use the update position function (or similar code) to change motor values.  Examples in the comments of the motor_bus header explain that code
                    
                if ch == 'a':
                    print("\r\nPan right")
                    interface.pan_arm(-30)
                    continue
                    
                if ch == 'd':
                    print("\r\nPan left")
                    interface.pan_arm(30)
                    continue
                
                if ch == 'w':
                    print("\r\nMove up")
                    interface.extend_shoulder(80)
                    continue
                    
                if ch == 's':
                    print("\r\nMove down")
                    interface.extend_shoulder(-80)
                    continue
                
                if ch == 'y':
                    print("\r\nMove up")
                    interface.extend_elbow(-60)
                    continue
                    
                if ch == 'h':
                    print("\r\nMove down")
                    interface.extend_elbow(60)
                    continue
                
                if ch == 'j':
                    print("\r\nWrist right")
                    interface.twist_wrist(60)
                    continue
                
                if ch == 'l':
                    print("\r\nWrist left")
                    interface.twist_wrist(-60)
                    continue
                
                if ch == 'i':
                    print("\r\nWrist up")
                    interface.flex_wrist(-60)
                    continue
                
                if ch == 'k':
                    print("\r\nWrist down")
                    interface.flex_wrist(60)
                    continue
                
                if ch == "q":
                    print("\r\nOpening hand")
                    interface.hand_control(15)
                    continue
                
                if ch == "e":
                    print("\r\nClosing hand")
                    interface.hand_control(-15)
                    continue

                handled = False
                # find mapping
                for (posk, negk), jidx in DEFAULT_KEYMAP.items():
                    if ch == posk:
                        joints[jidx] += step
                        handled = True
                        break
                    if ch == negk:
                        joints[jidx] -= step
                        handled = True
                        break

                if not handled:
                    print(f"Unknown key: {repr(ch)}")
                    continue

                # clamp to sensible bounds (0..100)
                #for i in range(len(joints)):
                 #   joints[i] = max(0.0, min(100.0, joints[i]))

                # send to robot (best-effort

                print(f"Sent joints: {[round(x,2) for x in joints]} (step {step:.2f})")

    except KeyboardInterrupt:
        print("Interrupted — exiting")
    finally:
        interface.cleanup()
        print("Exiting teleop.")

if __name__ == "__main__":
    main()
