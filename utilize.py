import pybullet as p
import pybullet_utils.bullet_client as bc
import numpy as np
import os
from typing import Callable





def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def connect_pybullet(timestep, show_gui=False):
    """
    Create a pyullet instance with set physics params.
    """
    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        # pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        # 不展示GUI的套件
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        # 禁用 tinyrenderer 
        # pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, 0)
        # pb.configureDebugVisualizer(pb.COV_ENABLE_Y_AXIS_UP, 1)
        
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)

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


def set_debug_camera(pb, debug_camera_params):
    pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    pb.resetDebugVisualizerCamera(
        debug_camera_params['dist'],
        debug_camera_params['yaw'],
        debug_camera_params['pitch'],
        debug_camera_params['pos']
    )
def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, ord=2,axis=-1)

class Camera:
    def __init__(self, pb, debug_camera_params):
        self.width, self.height = debug_camera_params['image_size']
        self.near, self.far = debug_camera_params['near_val'], debug_camera_params['far_val']
        self.fov = debug_camera_params['fov']
        self._pb = pb
        cam_pitch = 0

        aspect = self.width / self.height
        self.view_matrix = self._pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=debug_camera_params['pos'], 
                                                                      distance = debug_camera_params['dist'], 
                                                                      yaw = debug_camera_params['yaw'], 
                                                                      pitch = debug_camera_params['pitch'],
                                                                      roll = cam_pitch, 
                                                                      upAxisIndex=2) # either 1 for Y or 2 for Z axis up.
        self.projection_matrix = self._pb.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = self._pb.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        return rgb, depth, seg

    def rgbd_2_world_batch(self, depth):
        # reference: https://stackoverflow.com/a/62247245
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        # print(position)

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)
    