import os
import os.path as op
import pickle
import json
import time
import torch
import numpy as np
import cv2
from utils.metric_logger import AverageMeter
from utils.read import save_mesh
from utils.geometric_layers import orthographic_projection
from utils.comm import is_main_process
from utils.miscellaneous import mkdir
from utils.renderer import Renderer
from utils.vis import visual_mesh, visual_skeleton
from model.loss import keypoint_2d_loss, keypoint_3d_loss, vertices_loss, edge_length_loss, normal_loss
import datetime
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

def save_checkpoint(model, args, epoch, iteration, logger, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoints','checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

class Runner(object):
    def __init__(self, args, model, mano_model):
        super(Runner, self).__init__()
        self.args = args
        self.model = model
        self.mano = mano_model
        self.renderer = Renderer(faces=mano_model.face)

    def train(self, dataloader, optimizer, scheduler, board, logger):
        args = self.args
        max_iter = len(dataloader)
        iters_per_epoch = max_iter // args.epochs
        start_training_time = time.time()
        end = time.time()
        criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
        criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
        criterion_vertices = torch.nn.L1Loss().cuda(args.device)
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        log_losses = AverageMeter()
        log_loss_2djoints = AverageMeter()
        log_loss_3djoints = AverageMeter()
        log_loss_vertices = AverageMeter()

        for iteration, (img_keys, images, annotations) in enumerate(dataloader):
            iteration += 1
            epoch = iteration // iters_per_epoch
            images = images.cuda()
            batch_size = images.size(0)
            data_time.update(time.time() - end)
            gt_2d_joints = annotations['joints_2d'].cuda()
            gt_pose = annotations['pose'].cuda()
            gt_betas = annotations['betas'].cuda()
            has_mesh = annotations['has_smpl'].cuda()
            has_3d_joints = has_mesh
            has_2d_joints = has_mesh
            # get gt 3D mesh and joints in MANO space
            gt_vertices, gt_3d_joints = self.mano.layer(gt_pose, gt_betas)
            gt_vertices = gt_vertices / 1000.0
            gt_3d_joints = gt_3d_joints / 1000.0
            # normalize gt based on hand's wrist 
            gt_3d_root = gt_3d_joints[:,self.mano.joints_name.index('Wrist'),:]
            gt_vertices = gt_vertices - gt_3d_root[:, None, :]
            gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
            gt_3d_joints_with_tag = torch.ones((batch_size,gt_3d_joints.shape[1],4)).cuda()
            gt_3d_joints_with_tag[:,:,:3] = gt_3d_joints
            # forward pass
            out = self.model(images)
            pred_camera = out['pred_camera']
            pred_vertices = out['pred_vertices']
            # use mano and predicted camera parameters to get 3d joint and 2d joint
            pred_3d_joints_from_mesh = self.mano.get_3d_joints(pred_vertices)
            pred_3d_root = pred_3d_joints_from_mesh[:,self.mano.joints_name.index('Wrist'),:]
            pred_vertices = pred_vertices - pred_3d_root[:, None, :]
            pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_root[:, None, :]
            pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
            loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_mesh, gt_3d_joints_with_tag, has_3d_joints)
            loss_vertices = vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_mesh)
            loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_mesh, gt_2d_joints, has_2d_joints)
            loss_edge = edge_length_loss(pred_vertices, gt_vertices, self.mano.face)
            loss_normal = normal_loss(pred_vertices, gt_vertices, self.mano.face)
            loss = args.joint_2d_loss_weight*loss_2d_joints + \
                   args.vertices_loss_weight*loss_vertices + \
                   args.joint_3d_loss_weight*loss_3d_joints + \
                   args.edge_loss_weight * loss_edge + \
                   args.normal_loss_weight * loss_normal
            # add to tensorboard
            board.add_scalar('loss', loss.item(), iteration)
            board.add_scalar('loss_2d_joints', loss_2d_joints.item(), iteration)
            board.add_scalar('loss_3d_joints', loss_3d_joints.item(), iteration)
            board.add_scalar('loss_vertices', loss_vertices.item(), iteration)
            board.add_scalar('loss_edge', loss_edge.item(), iteration)
            board.add_scalar('loss_normal', loss_normal.item(), iteration)
            # update logs
            log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
            log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
            log_loss_vertices.update(loss_vertices.item(), batch_size)
            log_losses.update(loss.item(), batch_size)
            # optimize network
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if iteration % args.logging_steps == 0 or iteration == max_iter:
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    ' '.join(
                    ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                    ).format(eta=eta_string, ep=epoch, iter=iteration, 
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                    + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                        log_losses.avg, log_loss_2djoints.avg, log_loss_3djoints.avg, log_loss_vertices.avg, batch_time.avg, data_time.avg, 
                        optimizer.param_groups[0]['lr'])
                )


            if iteration % iters_per_epoch == 0:
                # checkpoint_dir = save_checkpoint(self.model, args, epoch, iteration, logger)
                if epoch%10==0:
                    checkpoint_dir = save_checkpoint(self.model, args, epoch, iteration, logger)
        
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info('Total training time: {} ({:.4f} s / iter)'.format(
            total_time_str, total_training_time / max_iter)
        )
        checkpoint_dir = save_checkpoint(self.model, args, epoch, iteration, logger)
              

    def evaluation(self, dataloader):
        self.model.eval()
        mesh_output_save = []
        joint_output_save = []
        with torch.no_grad():
            for i, (img_keys, images, annotations) in enumerate(tqdm(dataloader)):
                images = images.to(self.args.device)
                batch_size = images.shape[0]
                out = self.model(images)
                pred_vertices = out['pred_vertices']
                pred_3d_joints_from_mesh = self.mano.get_3d_joints(pred_vertices)
                for j in range(batch_size):
                    pred_vertices_list = pred_vertices[j].tolist()
                    mesh_output_save.append(pred_vertices_list)
                    pred_3d_joints_from_mesh_list = pred_3d_joints_from_mesh[j].tolist()
                    joint_output_save.append(pred_3d_joints_from_mesh_list)
        print('save results to pred.json')
        output_json_file = op.join(self.args.work_dir, 'out', 'eval', 'pred.json')
        print('save results to ', output_json_file)
        with open(output_json_file, 'w') as f:
            json.dump([joint_output_save, mesh_output_save], f)

        return    

    def demo(self, img_list):
        transform = transforms.Compose([           
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        transform_visualize = transforms.Compose([           
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()])
        self.model.eval()
        for img_file in img_list:
            img_file = op.join(self.args.work_dir, 'images', img_file)
            img = Image.open(img_file)
            img_tensor = transform(img)
            img_visual = transform_visualize(img).numpy().transpose(1, 2, 0)
            batch_img = img_tensor.unsqueeze(0).to(self.args.device)
            out = self.model(batch_img)
            pred_camera = out['pred_camera']
            pred_vertices = out['pred_vertices']
            pred_3d_joints_from_mesh = self.mano.get_3d_joints(pred_vertices)
            pred_3d_pelvis = pred_3d_joints_from_mesh[:, self.mano.joints_name.index('Wrist'), :]
            pred_vertices -= pred_3d_pelvis
            pred_3d_joints_from_mesh -= pred_3d_pelvis
            pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
            # visualize result
            skl_img = visual_skeleton(img_visual, pred_2d_joints_from_mesh[0])
            # rend_img = visual_mesh(self.renderer, img_visual, pred_vertices[0], pred_camera[0])
            # result_img = np.hstack([img_visual, skl_img, rend_img])[:,:,::-1] * 255
            result_img = np.hstack([img_visual, skl_img, ])[:,:,::-1] * 255
            img_save_path = op.join(self.args.output_dir, 'demo', op.basename(img_file))
            cv2.imwrite(img_save_path, result_img)
            save_mesh(img_save_path[:-4]+'.ply', pred_vertices[0].detach().cpu().numpy(), self.mano.face)
            print('save to ' + img_save_path)





            


