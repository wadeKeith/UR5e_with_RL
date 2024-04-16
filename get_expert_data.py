import numpy as np
from env import UR5Env
import random
import numpy as np
from sac_her import SACContinuous, ReplayBuffer_Trajectory, Trajectory,Agent_test
import math
import torch




















if __name__ == '__main__':
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
            'far_val': 5.0,
            'show_vision': False
        }
    robot_params = {
        "reset_arm_poses": reset_arm_poses,
        "reset_gripper_range": reset_gripper_range,
    }
    # control type: joint, end
    sim_params = {"use_gui":False,
                'timestep':1/240,
                'control_type':'end',
                'gripper_enable':False,
                'is_train':True,
                'distance_threshold':0.05,}
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = UR5Env(sim_params, robot_params,visual_sensor_params)