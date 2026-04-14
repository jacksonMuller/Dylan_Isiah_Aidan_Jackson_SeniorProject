import arm_interface, so101_forward_kinematics
import time
import sys
import math

def rad_to_deg(radians):
        return float(radians) * (180/math.pi)

def main(degrees): # Point shoulder pan to the degrees value given from the command line
    arm = arm_interface.RobotMotorInterface(motors=arm_interface.DEFAULT_MOTORS_DEGREES)
    clamped = float(max(min(degrees, 75), -75))
    print(f"Going to angle {clamped}")
    arm.bus.write("Goal_Position", "shoulder_pan", clamped, normalize=True)
    time.sleep(3)
    arm.cleanup()
    
def full_pos(radians): # Full position based on command line arguments in the form of radians
    arm = arm_interface.RobotMotorInterface(motors=arm_interface.DEFAULT_MOTORS_DEGREES)
    position_dict = {}
    for i in range(len(radians) - 1):
        position_dict[arm.joint_names[i+1]] = radians[i]
    position_dict["wrist_flex"] = 0.0
    position_dict["gripper"] = -62.0
    position_dict["wrist_roll"] = 3.2
    print(f"Sending positions: {position_dict}")
    arm.move_to_pose(position_dict, duration=0)
    time.sleep(5)
    arm.cleanup()

if __name__ == "__main__":
    print(sys.argv)
    radians = [rad_to_deg(arg) for arg in sys.argv[1:]]
    full_pos(radians) # Change this between main and full_pos
