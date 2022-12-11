import torch.nn.functional as F
import torch
import numpy as np

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0,:]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0,:]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()

def edge_length_loss(pred, gt, face, is_valid=None):
    d1_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_out = torch.sqrt(torch.sum((pred[:, face[:, 1], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    d1_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_gt = torch.sqrt(torch.sum((gt[:, face[:, 1], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    # valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
    # valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
    # valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

    diff1 = torch.abs(d1_out - d1_gt)  # * valid_mask_1
    diff2 = torch.abs(d2_out - d2_gt)  # * valid_mask_2
    diff3 = torch.abs(d3_out - d3_gt)  # * valid_mask_3
    loss = torch.cat((diff1, diff2, diff3), 1)
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()
