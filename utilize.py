import pybullet as p
import pybullet_utils.bullet_client as bc
import pkgutil
from importlib import util
import numpy as np
import os

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


def set_debug_camera(pb, debug_camera_params):
    pb.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    pb.resetDebugVisualizerCamera(
        debug_camera_params['dist'],
        debug_camera_params['yaw'],
        debug_camera_params['pitch'],
        debug_camera_params['pos']
    )