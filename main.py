import pybullet as p
import pybullet_utils.bullet_client as bc
import time
from importlib import util
from pprint import pprint
from utilize import connect_pybullet, load_standard_environment, set_debug_camera
import numpy as np
from Arm import ArmEmbodiment


root_path = '/Users/yin/Documents/GitHub/robotics_pybullet_learn/UR5'
timestep = 1/240
rest_poses = np.array([0.0, 0.315, -1.401, -2.401, -0.908, 1.570, 3.461, 0.0, 0.0, 0.0, 0.0, 0.0])
visual_sensor_params = {
        'image_size': [128, 128],
        'dist': 1.0,
        'yaw': 90.0,
        'pitch': -25.0,
        'pos': [0.6, 0.0, 0.0525],
        'fov': 75.0,
        'near_val': 0.1,
        'far_val': 100.0,
        'show_vision': False
    }
# 连接物理引擎
use_gui = False

pb = connect_pybullet(timestep, show_gui=use_gui)
robot_arm_params = {
    "rest_poses": rest_poses,
}
load_standard_environment(pb,root_path)
embodiment = ArmEmbodiment(pb, robot_arm_params=robot_arm_params)
set_debug_camera(pb, visual_sensor_params)



# available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

# pprint([p.getJointInfo(robot_id, i)[1] for i in available_joints_indexes])


while pb.isConnected():
    pb.stepSimulation()
    time.sleep(timestep)
    q_key = ord("q")
    keys = pb.getKeyboardEvents()
    if q_key in keys and keys[q_key] & pb.KEY_WAS_TRIGGERED:
        exit()

# # 断开连接
p.disconnect()