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
robot_params = {
    "reset_arm_poses": reset_arm_poses,
    "reset_gripper_range": reset_gripper_range,
}



use_gui = True
env = Env(use_gui, timestep, robot_params,visual_sensor_params)
obs = env.reset()





while True:
    env.step_simulation()
    # time.sleep(timestep)
    q_key = ord("q")
    keys = env._pb.getKeyboardEvents()
    if q_key in keys and keys[q_key] & env._pb.KEY_WAS_TRIGGERED:
        exit()

# # 断开连接
env.close()