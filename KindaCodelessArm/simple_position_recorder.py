import argparse
import json
import time
from pathlib import Path
from lerobot.motors.feetech import FeetechMotorsBus, TorqueMode
#from lerobot.motors.feetech.config import FeetechMotorsBusConfig


class SimplePositionSequencer:
    def __init__(self, port="COM15"):
        self.port = port
        self.motor_bus = None
        self.motors_config = {
            "shoulder_pan": [1, "sts3215"],
            "shoulder_lift": [2, "sts3215"],
            "elbow_flex": [3, "sts3215"],
            "wrist_flex": [4, "sts3215"],
            "wrist_roll": [5, "sts3215"],
            "gripper": [6, "sts3215"],
        }
        
    def connect(self):
        """Connect and completely disable ALL limits"""
        print(f"Connecting to robot on {self.port}...")
        config = FeetechMotorsBusConfig(port=self.port, motors=self.motors_config)
        self.motor_bus = FeetechMotorsBus(config)
        self.motor_bus.connect()
        
        # COMPLETELY REMOVE ALL LIMITS
        print("REMOVING ALL LIMITS...")
        for motor_name in self.motors_config.keys():
            try:
                self.motor_bus.write("Min_Angle_Limit", 0, motor_name)
                self.motor_bus.write("Max_Angle_Limit", 4095, motor_name)
                print(f"   {motor_name}: Limits removed")
            except Exception as e:
                print(f"   {motor_name}: {e}")
        
        print("Connected and ALL limits removed")
        
    def disconnect(self):
        if self.motor_bus:
            self.motor_bus.disconnect()
            print("Disconnected")
    
    def torque_off(self):
        """Disable torque for manual movement"""
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)
        print("Torque OFF - move robot freely")
    
    def torque_on(self):
        """Enable torque for controlled movement"""
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("Torque ON - robot under control")
    
    def get_positions(self):
        """Get current positions"""
        positions = {}
        for motor_name in self.motors_config.keys():
            pos = self.motor_bus.read("Present_Position", motor_name)
            if hasattr(pos, '__len__') and len(pos) == 1:
                pos = pos[0]
            positions[motor_name] = int(pos)
        return positions
    
    def move_to_position(self, positions, duration_seconds):
        """Move to position with timing - SMOOTH MOVEMENTS"""
        # Adjust timing for much smoother, slower movements
        if duration_seconds == 0:
            time_ms = 800  # Keep fast movements quick but not too fast
        else:
            # For timed movements, make them MUCH slower and smoother
            time_ms = int(duration_seconds * 1000)  
        
        for motor_name, position in positions.items():
            self.motor_bus.write("Goal_Time", time_ms, motor_name)
            self.motor_bus.write("Goal_Position", int(position), motor_name)
    
    def record_sequence(self, name):
        """Record a position sequence"""
        print(f"\n{'='*60}")
        print(f"RECORDING SEQUENCE: {name}")
        print(f"{'='*60}")
        print("INSTRUCTIONS:")
        print("1. Move robot to STARTING position")
        print("2. Press ENTER to record starting position")
        print("3. Move to next position")
        print("4. Enter seconds to reach position (0=fast, decimals OK)")
        print("5. Press ENTER to record position")
        print("6. Repeat steps 3-5 for each position")
        print("7. Press CTRL+C when done")
        
        self.torque_off()
        
        sequence = []
        position_num = 1
        
        try:
            # Starting position
            input(f"\nMove to STARTING position, press ENTER...")
            pos = self.get_positions()
            pos_str = " | ".join([f"{motor}:{position:4d}" for motor, position in pos.items()])
            print(f"START: {pos_str}")
            
            sequence.append({
                "position": position_num,
                "positions": pos.copy(),
                "duration": 0.0  # Starting position has no timing
            })
            position_num += 1
            
            # Subsequent positions
            while True:
                print(f"\nMove to position {position_num}")
                duration_input = input(f"Enter seconds to reach position {position_num} (0=fast): ").strip()
                
                try:
                    duration = float(duration_input)
                    if duration < 0:
                        print("Duration cannot be negative")
                        continue
                except ValueError:
                    print("Invalid number")
                    continue
                
                # Record position
                pos = self.get_positions()
                pos_str = " | ".join([f"{motor}:{position:4d}" for motor, position in pos.items()])
                time_desc = "FAST" if duration == 0 else f"{duration}s"
                print(f"Position {position_num}: {pos_str} ({time_desc})")
                
                sequence.append({
                    "position": position_num,
                    "positions": pos.copy(),
                    "duration": duration
                })
                position_num += 1
                
        except KeyboardInterrupt:
            print(f"\nRecording finished! Captured {len(sequence)} positions")
        
        # AUTOMATICALLY ADD STARTING POSITION AS FINAL POSITION
        if len(sequence) > 1:  # Only if we have more than just the start position
            start_position = sequence[0]["positions"].copy()
            sequence.append({
                "position": position_num,
                "positions": start_position,
                "duration": 1.0  # 1 second to return to start
            })
            print(f"Added return to START position as final step (1.0s)")
        
        # Save sequence
        Path("sequences").mkdir(exist_ok=True)
        file_path = Path("sequences") / f"{name}.json"
        
        data = {
            "name": name,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_positions": len(sequence),
            "sequence": sequence
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved to: {file_path}")
        
        # Re-enable torque
        self.torque_on()
    
    def play_sequence(self, name):
        """Play back a recorded sequence"""
        file_path = Path("sequences") / f"{name}.json"
        
        if not file_path.exists():
            print(f"Sequence file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        sequence = data["sequence"]
        
        print(f"\n{'='*60}")
        print(f"PLAYING SEQUENCE: {name}")
        print(f"{'='*60}")
        print(f"Positions: {len(sequence)}")
        
        # Show what we're going to do
        for step in sequence:
            pos_str = " | ".join([f"{motor}:{position:4d}" for motor, position in step["positions"].items()])
            duration = step["duration"]
            time_desc = "START" if duration == 0 and step["position"] == 1 else ("FAST" if duration == 0 else f"{duration}s")
            print(f"   {step['position']}. {pos_str} ({time_desc})")
        
        input("\nPress ENTER to start playback...")
        
        self.torque_on()
        
        # REMOVE LIMITS AGAIN before playback
        print("Ensuring limits are removed for playback...")
        for motor_name in self.motors_config.keys():
            try:
                self.motor_bus.write("Min_Angle_Limit", 0, motor_name)
                self.motor_bus.write("Max_Angle_Limit", 4095, motor_name)
            except:
                pass
        
        print(f"\nEXECUTING...")
        print("=" * 50)
        
        try:
            for i, step in enumerate(sequence):
                pos = step["positions"]
                duration = step["duration"]
                position_num = step["position"]
                
                pos_str = " | ".join([f"{motor}:{position:4d}" for motor, position in pos.items()])
                
                if position_num == 1:
                    # Go to starting position instantly
                    print(f"Position {position_num}: {pos_str} (moving to start)")
                    self.move_to_position(pos, 0)
                    time.sleep(1.2)  # Give a bit more time to reach start
                else:
                    if duration == 0:
                        print(f"Position {position_num}: {pos_str} (FAST)")
                        self.move_to_position(pos, 0)
                        time.sleep(1.0)  # Wait for fast movement 
                    else:
                        print(f"Position {position_num}: {pos_str} ({duration}s)")
                        self.move_to_position(pos, duration)
                        # Wait for the actual movement time (4x longer) plus buffer
                        actual_time = duration * 4.0
                        time.sleep(actual_time + 1.0)  # Wait for movement + larger buffer
            
            print(f"\nSEQUENCE COMPLETE!")
            
        except KeyboardInterrupt:
            print(f"\nPLAYBACK STOPPED")


def main():
    parser = argparse.ArgumentParser(description="Simple Position Sequence Recorder")
    parser.add_argument("--mode", choices=["record", "play"], required=True)
    parser.add_argument("--name", required=True, help="Sequence name")
    parser.add_argument("--port", default="COM15", help="Serial port")
    
    args = parser.parse_args()
    
    sequencer = SimplePositionSequencer(args.port)
    
    try:
        sequencer.connect()
        
        if args.mode == "record":
            sequencer.record_sequence(args.name)
        elif args.mode == "play":
            sequencer.play_sequence(args.name)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        sequencer.disconnect()
    
    return 0


if __name__ == "__main__":
    exit(main())
