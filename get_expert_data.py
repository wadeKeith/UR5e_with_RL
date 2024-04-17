import numpy as np
from reach_env import Reach_UR5Env
import random
import numpy as np
from sac_her import SACContinuous, ReplayBuffer_Trajectory, Trajectory,Agent_test
import math
import torch
import pickle
import tqdm


seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
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

expert_data_num = 10000
buffer_size = 100000
batch_size = 512
env = Reach_UR5Env(sim_params, robot_params,visual_sensor_params)
state_len = env.observation_space['observation'].shape[0]
achieved_goal_len = env.observation_space['achieved_goal'].shape[0]
desired_goal_len = env.observation_space['desired_goal'].shape[0]
her_buffer = ReplayBuffer_Trajectory(capacity= buffer_size, 
                                    dis_threshold=sim_params['distance_threshold'], 
                                    use_her=True,
                                    batch_size=batch_size,
                                    state_len=state_len,
                                    achieved_goal_len=achieved_goal_len,)

savetime = 0
for epoch in range(100000):
    if savetime >= expert_data_num:
        break
    # reset the environment
    observation, _ = env.reset()
    traj = Trajectory(observation.copy())
    done = False
    # start to collect samples
    # step_time = 0
    while not done:
        # step_time += 1
        # obs = observation[:state_len]
        achieved_goal = observation[state_len:state_len+achieved_goal_len]
        desired_goal = observation[state_len+achieved_goal_len:]
        # ee_pos = obs[int(state_len/2):]
        finger_pos = achieved_goal.copy()
        # print('delta',(desired_goal-finger_pos)*env.observation_space['desired_goal'].high[0])
        dx = np.clip((desired_goal[0]-finger_pos[0])*env.observation_space['desired_goal'].high[0]/env.arm_gripper.action_scale,-1,1)
        dy = np.clip((desired_goal[1]-finger_pos[1])*env.observation_space['desired_goal'].high[0]/env.arm_gripper.action_scale,-1,1)
        dz = np.clip((desired_goal[2]-finger_pos[2])*env.observation_space['desired_goal'].high[0]/env.arm_gripper.action_scale,-1,1)
        action = np.array([dx,dy,dz])
        # print('action:',action)
        observation,reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        traj.store_step(action.copy(), observation.copy(), reward, done)
    print(epoch)
    if info['is_success'] == True:
        savetime += 1
        print("This is " + str(savetime) + " savetime ")
        her_buffer.add_trajectory(traj)
file_name = "ur5_reach_"+str(savetime)+"_expert_data.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(her_buffer, file)