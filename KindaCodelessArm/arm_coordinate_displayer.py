from utils import arm_interface, so101_forward_kinematics

def main():
    arm = arm_interface.RobotMotorInterface()
    
    for _ in range(100):
        motor_pos = arm.bus.read("Present_Position", normalize=False)
        coord_pos = so101_forward_kinematics.get_forward_kinematics(motor_pos)
        print(f"Coord: {coord_pos} maps to motor positions: {motor_pos}")
    
    arm.cleanup()