from utils import arm_interface, so101_forward_kinematics
import time

def main():
    arm = arm_interface.RobotMotorInterface(motors=arm_interface.DEFAULT_MOTORS_DEGREES)
    motors = list(arm_interface.DEFAULT_MOTORS.keys())
    print(f"Displaying positions for the following motors: {motors}")
    for _ in range(100):
        
        motor_pos = arm.bus.sync_read("Present_Position", motors, normalize=True)
        raw_pos = arm.bus.read("Present_Position", "elbow_flex", normalize=False)
        print(f"Raw elbow flex position: {raw_pos}")
        print(f"Motor positions: {motor_pos}")
        coord_pos, _ = so101_forward_kinematics.get_forward_kinematics(motor_pos)
        print(f"Coord: {coord_pos}\n====================")
        time.sleep(1)
    
    arm.cleanup()

if __name__ == "__main__":
    main()
