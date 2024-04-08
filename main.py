import pybullet as p
import pybullet_utils.bullet_client as bc
import time
from importlib import util
from pprint import pprint
from utilize import connect_pybullet, set_debug_camera
import numpy as np
from env import Env
import math


root_path = '/Users/yin/Documents/GitHub/robotics_pybullet_learn/UR5'
timestep = 1/240
reset_arm_poses = [math.pi, -math.pi/2, -math.pi*5/9, -math.pi*4/9,
                               math.pi/2, 0]
reset_gripper_range = [0, 0.085]
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
use_gui = True

pb = connect_pybullet(timestep, show_gui=use_gui)
robot_params = {
    "reset_arm_poses": reset_arm_poses,
    "reset_gripper_range": reset_gripper_range,
}
env = Env(pb, robot_params=robot_params,root_path=root_path)
env.reset()
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