import numpy as np
from pick_place_env import PickPlace_UR5Env
import random
import numpy as np
from sac_her import SACContinuous, ReplayBuffer_Trajectory, Trajectory,Agent_test
import math
import torch
import pickle
import tqdm
from utilize import distance


seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
reset_arm_poses = [0, -math.pi/2, math.pi*4/9, -math.pi*4/9,
                            -math.pi/2, 0]
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
            'gripper_enable':True,
            'is_train':True,
            'distance_threshold':0.05,}

expert_data_num = 10000
buffer_size = 100000
batch_size = 512
env = PickPlace_UR5Env(sim_params, robot_params,visual_sensor_params)
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
for epoch in range(1000000):
    if savetime >= expert_data_num:
        break
    # reset the environment
    obs_norm, _,obs_dict = env.reset()
    traj = Trajectory(obs_norm.copy())
    done = False
    # start to collect samples
    step_time = 0
    while not done:
        
        obs = obs_dict['observation']
        achieved_goal = obs_dict['achieved_goal']
        desired_goal = obs_dict['desired_goal']
        # ee_pos = obs[int(state_len/2):]
        finger_pos = achieved_goal[:3]
        block_pos = achieved_goal[3:]
        # print('delta',(achieved_goal-finger_pos))
        if distance(block_pos, finger_pos)>sim_params['distance_threshold'] and distance(achieved_goal,desired_goal)>sim_params['distance_threshold']:
            dx = np.clip((block_pos[0]-finger_pos[0]),-1,1)
            dy = np.clip((block_pos[1]-finger_pos[1]),-1,1)
            dz = np.clip((block_pos[2]-finger_pos[2])+0.04,-1,1)
            d_gripper = 0
        elif distance(block_pos,finger_pos)<=sim_params['distance_threshold'] and distance(achieved_goal,desired_goal)>sim_params['distance_threshold']:
            if step_time <10:
                dx = np.clip((block_pos[0]-finger_pos[0]),-1,1)
                dy = np.clip((block_pos[1]-finger_pos[1]),-1,1)
                dz = np.clip((block_pos[2]-finger_pos[2])+0.04,-1,1)
                d_gripper = -1
            else:
                dx = np.clip((desired_goal[3]-block_pos[0]),-1,1)
                dy = np.clip((desired_goal[4]-block_pos[1]),-1,1)
                dz = np.clip((desired_goal[5]-block_pos[2]),-1,1)
                d_gripper = -1
            step_time += 1
        action = np.array([dx,dy,dz,d_gripper])
        # print('action:',action)
        obs_norm,reward, terminated, truncated, info,obs_dict = env.step(action)
        done = terminated or truncated
        traj.store_step(action.copy(), obs_norm.copy(), reward, done)
    print(epoch)
    if info['is_success'] == True:
        savetime += 1
        print("This is " + str(savetime) + " savetime ")
        her_buffer.add_trajectory(traj)
file_name = "ur5_pickplace_"+str(savetime)+"_expert_data.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(her_buffer, file)