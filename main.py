import pybullet as p
import pybullet_utils.bullet_client as bc
import time
from importlib import util
from pprint import pprint
import os
import numpy as np
from Arm import ArmEmbodiment

def connect_pybullet(timestep, show_gui=False):
    """
    Create a pyullet instance with set physics params.
    """
    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        egl = util.find_spec("eglRenderer")
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    pb.setGravity(0, 0, -9.81)
    pb.setPhysicsEngineParameter(
        fixedTimeStep=timestep,
        numSolverIterations=300,
        numSubSteps=1,
        contactBreakingThreshold=0.0005,
        erp=0.05,
        contactERP=0.05,
        frictionERP=0.2,
        solverResidualThreshold=1e-7,
        contactSlop=0.001,
        globalCFM=0.0001,
    )
    return pb
def load_standard_environment(pb,root_path):
    """
    Load a standard environment with a plane and a table.
    """
    pb.loadURDF(
        os.path.join(root_path,"shared_assets/environment_objects/plane/plane.urdf"),
        [0, 0, -0.625],
    )
    pb.loadURDF(
        os.path.join(root_path,"shared_assets/environment_objects/table/table.urdf"),
        [0.50, 0.00, -0.625],
        [0.0, 0.0, 0.0, 1.0],
)

root_path = '/Users/yin/Documents/GitHub/robotics_pybullet_learn/UR5'
timestep = 1/240
rest_poses = np.array([0.0, 0.315, -1.401, -2.401, -0.908, 1.570, 3.461, 0.0, 0.0, 0.0, 0.0, 0.0])
# 连接物理引擎
use_gui = True

pb = connect_pybullet(timestep, show_gui=use_gui)
robot_arm_params = {
    "rest_poses": rest_poses,
}
load_standard_environment(pb,root_path)
embodiment = ArmEmbodiment(pb, robot_arm_params=robot_arm_params)



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