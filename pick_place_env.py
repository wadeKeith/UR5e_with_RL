from typing import Any, Dict, Optional, Tuple
import time
from ur5_robotiq import UR5Robotiq140
from utilize import connect_pybullet, set_debug_camera, Camera, distance
from gymnasium import spaces
import numpy as np
import math


class PickPlace_UR5Env(object):
    def __init__(self, sim_params, robot_params, visual_sensor_params):
        # super().__init__()
        self.vis = sim_params['use_gui']
        self._pb = connect_pybullet(sim_params['timestep'], show_gui=self.vis)
        self.SIMULATION_STEP_DELAY = sim_params['timestep']
        self.control_type = sim_params['control_type']
        self.gripper_enable = sim_params['gripper_enable']
        self.is_train = sim_params['is_train']
        self.distance_threshold = sim_params['distance_threshold']
        self.load_standard_environment()

        # initialize a robot arm and gripper
        self.arm_gripper = UR5Robotiq140(
            self._pb,
            robot_params=robot_params,
            use_gui = self.vis,
        )
        self.arm_gripper.step_simulation = self.step_simulation
        # # Initialize the camera
        # self.camera = Camera(self._pb, visual_sensor_params)
        if self.vis:
            set_debug_camera(self._pb, visual_sensor_params)
        # Initialize the goal range
        self.blockUid = -1
        self.goal_range_low = np.array([0.7, -0.3, 0.04])
        self.goal_range_high = np.array([0.8, 0.3, 0.1])
        # rgb_obs_space = spaces.Box(low=0, high=255, shape=(visual_sensor_params['image_size'][0], visual_sensor_params['image_size'][1], 4), dtype=np.uint8)
        # depth_obs_space = spaces.Box(low=0, high=1, shape=(visual_sensor_params['image_size'][0], visual_sensor_params['image_size'][1]), dtype=np.float32)
        # seg_obs_space = spaces.Box(low=-1, high=255, shape=(visual_sensor_params['image_size'][0], visual_sensor_params['image_size'][1]), dtype=np.int32)
        if self.control_type=='joint' and self.gripper_enable:
            observation_bound_now = np.ones(shape=(self.arm_gripper.num_control_dofs,))*3.14159265359
            observation_bound = np.concatenate([observation_bound_now,observation_bound_now])
        elif self.control_type=='joint' and self.gripper_enable==False:
            observation_bound_now = np.ones(shape=(self.arm_gripper.arm_num_dofs,))*3.14159265359
            observation_bound = np.concatenate([observation_bound_now,observation_bound_now])
        elif self.control_type=='end' and self.gripper_enable:
            observation_bound = np.concatenate([np.ones(shape=(3,))*2,
                                                np.ones(shape=(3,))*3.14159265359,
                                                np.ones(shape=(3,))*2,
                                                np.ones(shape=(3,))*3.2,
                                                np.ones(shape=(self.arm_gripper.num_control_dofs-self.arm_gripper.arm_num_dofs,))*3.14159265359,
                                                np.ones(shape=(3,))*3.14159265359,
                                                np.ones(shape=(3,))*2])
            # observation_bound = np.concatenate([observation_bound_now,observation_bound_now])
        else:
            observation_bound_now = np.array([2, 2, 2])
            observation_bound = np.concatenate([observation_bound_now,observation_bound_now])
        observation_space = spaces.Box(-observation_bound, observation_bound, dtype=np.float32)
        achieved_space = spaces.Box(np.array([0.7, -0.3, 0]), self.goal_range_high, dtype=np.float32)
        desired_space = spaces.Box(np.array([0.7, -0.3, 0]), self.goal_range_high, dtype=np.float32)
        # self.observation_space = spaces.Dict({
        #     'rgb': rgb_obs_space,
        #     'depth': depth_obs_space,
        #     'seg': seg_obs_space,
        #     'positions': positions_obs_space,
        #     'velocities': velocities_obs_space,
        #     'finger_pos': ee_pos_obs_space
        # })
        self.observation_space = spaces.Dict({
            'observation': observation_space,
            'achieved_goal': achieved_space,
            'desired_goal': desired_space,
        })
        n_action = 3 if self.control_type == "end" else 6  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 1 if self.gripper_enable else 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_action,),dtype=np.float32)
        self.time = None
        self.time_limitation = 200
        self.n_sub_step = 50

        

    

    def reset(self,seed=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the pose of the arm and sensor
        """
        self.time = 0
        self.goal = None
        self.goal_ang = None
        self.achieved_goal_initial = None
        self.achieved_goal_initial_ang = None
        
        self.arm_gripper.reset(self.gripper_enable)
        # block: x in (0.6, 1), y in (-0.4, 0.4), z = 0.04
        # target: x in (0.6, 1), y in (-0.4, 0.4), z in (0.04, 0.4)
        if self.is_train:
            initial_done = False
            while not initial_done:
                achieved_goal_initial,achieved_goal_initial_ang = self._sample_achieved_goal_initial()
                goal,goal_ang = self._sample_goal()
                initial_done = False if distance(achieved_goal_initial, goal) <= self.distance_threshold else True
            self.achieved_goal_initial = achieved_goal_initial.copy()
            self.achieved_goal_initial_ang = achieved_goal_initial_ang
            self.goal = goal.copy()
            self.goal_ang = goal_ang
        else:
            self.achieved_goal_initial = np.array([0.72,0.15,0.04])
            self.achieved_goal_initial_ang = 0
            self.goal = np.array([0.78,-0.15,0.1])
            self.goal_ang = 0
        if self.blockUid == -1:
            self.blockUid = self._pb.loadURDF("./assets/urdfs/cube_small_pick.urdf", self.achieved_goal_initial,
                                        self._pb.getQuaternionFromEuler([0,0,self.achieved_goal_initial_ang]))
            self.targetUid = self._pb.loadURDF("./assets/urdfs/cube_small_target_pick.urdf",
                                        self.goal,
                                        self._pb.getQuaternionFromEuler([0,0,self.goal_ang]), useFixedBase=1)
        else:
            self._pb.removeBody(self.blockUid)
            self._pb.removeBody(self.targetUid)
            self.blockUid = self._pb.loadURDF("./assets/urdfs/cube_small_pick.urdf", self.achieved_goal_initial,
                                        self._pb.getQuaternionFromEuler([0,0,self.achieved_goal_initial_ang]))
            self.targetUid = self._pb.loadURDF("./assets/urdfs/cube_small_target_pick.urdf",
                                        self.goal,
                                        self._pb.getQuaternionFromEuler([0,0,self.goal_ang]), useFixedBase=1)
        self._pb.setCollisionFilterPair(self.targetUid, self.blockUid, -1, -1, 0)
        robot_obs = self.arm_gripper.get_joint_obs(self.control_type,self.gripper_enable).copy()
        # robot_obs = np.concatenate([robot_obs_old, robot_obs_new])
        obs_dict = self._get_obs(robot_obs)
        obs = self.dictobs2npobs(obs_dict, self.observation_space)
        info = {"is_success": bool(self.is_success(obs_dict["achieved_goal"], obs_dict['desired_goal']))}
        return (obs, info,obs_dict)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        self.time +=1
        # robot_obs_old = self.arm_gripper.get_joint_obs(self.control_type,self.gripper_enable).copy() 
        assert self.control_type in ('joint', 'end')
        if self.gripper_enable:
            self.arm_gripper.move_ee(action[:-1], self.control_type)
            self.arm_gripper.move_gripper(action[-1])
        else:
            self.arm_gripper.move_ee(action, self.control_type)
        self.step_simulation()
        robot_obs = self.arm_gripper.get_joint_obs(self.control_type,self.gripper_enable).copy()
        # robot_obs = np.concatenate([robot_obs_old, robot_obs_new])
        obs_dict = self._get_obs(robot_obs)
        obs = self.dictobs2npobs(obs_dict, self.observation_space)
        info = {"is_success": bool(self.is_success(obs_dict['achieved_goal'], obs_dict['desired_goal']))}
        terminated = self.compute_terminated(obs_dict['achieved_goal'], obs_dict['desired_goal'], info)
        truncated = self.compute_truncated(obs_dict['achieved_goal'], obs_dict['desired_goal'], info)
        reward  = float(self.compute_reward(obs_dict['achieved_goal'], obs_dict['desired_goal'], info))
        
        return obs, reward, terminated, truncated, info,obs_dict

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return -np.array(d > self.distance_threshold, dtype=np.float32)
    
    def compute_terminated(self, achieved_goal, desired_goal, info) -> bool:
        d = distance(achieved_goal, desired_goal)
        return d <= self.distance_threshold
    
    def compute_truncated(self, achieved_goal, desired_goal, info) -> bool:
        d = distance(achieved_goal, desired_goal)
        if d <= self.distance_threshold:
            return True
        else:
            if self.time >=self.time_limitation:
                return True
            else:
                return False

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d <= self.distance_threshold, dtype=bool)
    
    
    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        for _ in range(self.n_sub_step):
            self._pb.stepSimulation()
            if self.vis:
                time.sleep(self.SIMULATION_STEP_DELAY)

    def load_standard_environment(self):
        """
        Load a standard environment with a plane and a table.
        """
        self._pb.loadURDF(
            "./assets/environment_objects/plane/plane.urdf",
            [0, 0, -0.625],
        )
        self._pb.loadURDF(
            "./assets/environment_objects/table/table.urdf",
            [0.50, 0.00, -0.625],
            [0.0, 0.0, 0.0, 1.0], globalScaling=1,
    )

    def _get_obs(self, robot_obs):
        achieved_goal,achieved_goal_orn = self.get_achieved_goal()
        desired_goal = self.goal.copy()
        relative_pos = achieved_goal-robot_obs[:3].copy()
        total_obs = np.concatenate([robot_obs.copy(),achieved_goal_orn.copy(),relative_pos.copy()])
        return {
            'observation': total_obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }
    def dictobs2npobs(self,observation, dic_observation_space):
        list_obs = []
        for key in observation.keys():
            list_obs += ((observation[key]-dic_observation_space[key].low)/(dic_observation_space[key].high-dic_observation_space[key].low)).tolist()
        return np.array(list_obs)


    def get_achieved_goal(self) -> np.ndarray:
        achieved_goal_pos,achieved_goal_orn_qua = self._pb.getBasePositionAndOrientation(self.blockUid)
        achieved_goal_orn = self._pb.getEulerFromQuaternion(achieved_goal_orn_qua)
        # self._pb.addUserDebugPoints(pointPositions = [achieved_goal_finger_pos], pointColorsRGB = [[0, 0, 255]], pointSize= 20, lifeTime= 0)
        return np.array(achieved_goal_pos), np.array(achieved_goal_orn)
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal_ang = np.random.uniform(-math.pi, math.pi)
        if np.random.random() < 0.3:
            goal[2] = self.goal_range_low[-1]
        return goal,goal_ang

    def _sample_achieved_goal_initial(self):
        achieved_goal_inital_xy = np.random.uniform(self.goal_range_low[:2], self.goal_range_high[:2])
        achieved_goal_inital = np.concatenate([achieved_goal_inital_xy,np.array(self.goal_range_low[-1]).reshape(1,)])
        achieved_goal_inital_ang = np.random.uniform(-math.pi, math.pi)
        return achieved_goal_inital,achieved_goal_inital_ang

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()
    def render(self, mode='human'):
        pass
    def seed(self, seed=None):
        pass
