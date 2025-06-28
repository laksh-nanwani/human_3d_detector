# human_3d_detector

### Dependencies
```
pip install ultralytics opencv-python numpy message_filters
sudo apt install ros-${ROS_DISTRO}-vision-msgs
sudo apt install ros-${ROS_DISTRO}-cv-bridge
```

### Build and Run
```
colcon build --packages-select human_3d_detector
source install/setup.bash

ros2 run human_3d_detector human_detector_node
```
