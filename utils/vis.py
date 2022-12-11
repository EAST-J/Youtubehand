import numpy as np
import torch
from utils.renderer import Renderer, visualize_reconstruction_no_text, draw_skeleton

def visual_mesh(renderer, images, pred_vertices, pred_camera):
    '''
    images: H * W * 3
    pred_vertices: v * 3
    pred_camera: 3
    '''
    vertices = pred_vertices.detach().cpu().numpy()
    pred_camera = pred_camera.detach().cpu().numpy()
    rend_img = visualize_reconstruction_no_text(images, images.shape[0], vertices, pred_camera, renderer, color='hand')
    return rend_img

def visual_skeleton(images, pred_2d_joints):
    '''
    images: H * W * 3
    pred_2d_joints: J * 2
    '''
    img_size = images.shape[0]
    pred_kp = pred_2d_joints.detach().cpu().numpy()
    pred_joint = ((pred_kp + 1) * 0.5) * img_size
    skel_img = draw_skeleton(images, pred_joint)
    return skel_img