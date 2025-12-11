# ARM Control Manual

**Project:** Vision-Guided Robotic Arm  
**Version:** 1.0  
**Authors:** Aidan Hodges, Dylan Suzuki, Isaiah Pajarillo, Jackson Muller

## Instructions for OpenCV

### 1: Connect or Install your Camera
Use a built-in webcaam or plug in a USB camera compatible with your Raspberry Pi

### 2: Install OpenCV here 
Download the latest release: https://opencv.org/releases/

### 3: Install Python on the Raspberry Pi
Ensure Python 3 is installed and updated

### 4: Create and Activate a virtual envrinment
Run the following commands:
python3 -m venv venv
source venv/bin/activate

### 5: Run the Object-Detection Script
Navigate to the KindaCodelessArm folder from this repo and run:
python3 object-ident.py


## Instructions for the arm

### 1: Obtain the #D-Printed Parts and Servos
All the #D models and hardwre info can be found here:
https://github.com/TheRobotStudio/SO-ARM100 

### 2: Assemble the arm
Follow the assembly instruction from the repo above

### 3: Connect the Arm to the Raspberry Pi
Plug the servos into the designated control board pins and ensure proper power delivery.

### 4: Install Python on the Raspberry Pi
Make sure Python is installed and accessible via the terminal

### 5: Run the Motor-Control Program
From the KindaCodelessArm folder, excecute:
python3 keyboard_motor_control.py


### 6: Refer to the controls found here ![ARM Control Manual](Images/ARM%20Control%20Manual.png)
