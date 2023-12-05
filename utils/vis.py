import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
import torch
import cv2
import pyrender
import trimesh

# TODO: 考虑还是将render的部分换成Pytorch3D?
class Visualizer():
    def __init__(self, img_size=224):
        self.img_size = img_size

    def draw_mesh(self, input_image, verts, faces, pred_camera):
        render = pyrender.OffscreenRenderer(viewport_width=self.img_size,
                                    viewport_height=self.img_size, point_size=1.0)
        pred_camera_t = torch.stack([pred_camera[1],
                              pred_camera[2],
                              2 * 5000. / (self.img_size * pred_camera[0] + 1e-9)], dim=-1)
        verts = verts + pred_camera_t
        verts = verts.detach().cpu().numpy()
        # transform the coordinates
        verts[:, 1:] *= -1
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3)
                           )
        focal_length = 5000.
        camera_center = [self.img_size / 2, self.img_size / 2]
        pyrender_cam = pyrender.IntrinsicsCamera(focal_length, focal_length, camera_center[0], camera_center[1])
        scene.add(pyrender_cam, pose=np.eye(4))
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            smooth=True,
            wireframe=True,
            roughnessFactor=1.0,
            emissiveFactor=(0.1, 0.1, 0.1),
            baseColorFactor=(1.0, 1.0, 0.9, 1.0)
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh, 'mesh')
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose)
        render_image, render_depth = render.render(scene, flags=pyrender.RenderFlags.RGBA)
        render_image = render_image / 255.
        valid_mask = (render_depth > 0)[:, :, np.newaxis]

        output_img = render_image * valid_mask + (1 - valid_mask) * input_image

        output_img = output_img
        return output_img

    def draw_skeleton(self, input_image, joints, draw_edges=True, vis=None, radius=None):

        """
        joints is 3 x 19. but if not will transpose it.
        0: Right ankle
        1: Right knee
        2: Right hip
        3: Left hip
        4: Left knee
        5: Left ankle
        6: Right wrist
        7: Right elbow
        8: Right shoulder
        9: Left shoulder
        10: Left elbow
        11: Left wrist
        12: Neck
        13: Head top
        14: nose
        15: left_eye
        16: right_eye
        17: left_ear
        18: right_ear
        """
        # normalize the joints
        joints = joints.detach().cpu().numpy()
        joints = ((joints + 1) * 0.5) * self.img_size

        if radius is None:
            radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

        colors = {
            'pink': (197, 27, 125),  # L lower leg
            'light_pink': (233, 163, 201),  # L upper leg
            'light_green': (161, 215, 106),  # L lower arm
            'green': (77, 146, 33),  # L upper arm
            'red': (215, 48, 39),  # head
            'light_red': (252, 146, 114),  # head
            'light_orange': (252, 141, 89),  # chest
            'purple': (118, 42, 131),  # R lower leg
            'light_purple': (175, 141, 195),  # R upper
            'light_blue': (145, 191, 219),  # R lower arm
            'blue': (69, 117, 180),  # R upper arm
            'gray': (130, 130, 130),  #
            'white': (255, 255, 255),  #
        }

        image = input_image.copy()
        input_is_float = False

        if np.issubdtype(image.dtype, np.float):
            input_is_float = True
            max_val = image.max()
            if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
                image = (image * 255).astype(np.uint8)
            else:
                image = (image).astype(np.uint8)

        if joints.shape[0] != 2:
            joints = joints.T
        joints = np.round(joints).astype(int)

        jcolors = [
            'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
            'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
            'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
            'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
        ]

        if joints.shape[1] == 19:
            # parent indices -1 means no parents
            parents = np.array([
                1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
            ])
            # Left is light and right is dark
            ecolors = {
                0: 'light_pink',
                1: 'light_pink',
                2: 'light_pink',
                3: 'pink',
                4: 'pink',
                5: 'pink',
                6: 'light_blue',
                7: 'light_blue',
                8: 'light_blue',
                9: 'blue',
                10: 'blue',
                11: 'blue',
                12: 'purple',
                17: 'light_green',
                18: 'light_green',
                14: 'purple'
            }
        elif joints.shape[1] == 14:
            parents = np.array([
                1,
                2,
                8,
                9,
                3,
                4,
                7,
                8,
                -1,
                -1,
                9,
                10,
                13,
                -1,
            ])
            ecolors = {
                0: 'light_pink',
                1: 'light_pink',
                2: 'light_pink',
                3: 'pink',
                4: 'pink',
                5: 'pink',
                6: 'light_blue',
                7: 'light_blue',
                10: 'light_blue',
                11: 'blue',
                12: 'purple'
            }
        elif joints.shape[1] == 21:  # hand
            parents = np.array([
                -1,
                0,
                1,
                2,
                3,
                0,
                5,
                6,
                7,
                0,
                9,
                10,
                11,
                0,
                13,
                14,
                15,
                0,
                17,
                18,
                19,
            ])
            ecolors = {
                0: 'light_purple',
                1: 'light_green',
                2: 'light_green',
                3: 'light_green',
                4: 'light_green',
                5: 'pink',
                6: 'pink',
                7: 'pink',
                8: 'pink',
                9: 'light_blue',
                10: 'light_blue',
                11: 'light_blue',
                12: 'light_blue',
                13: 'light_red',
                14: 'light_red',
                15: 'light_red',
                16: 'light_red',
                17: 'purple',
                18: 'purple',
                19: 'purple',
                20: 'purple',
            }
        else:
            print('Unknown skeleton!!')

        for child in range(len(parents)):
            point = joints[:, child]
            # If invisible skip
            if vis is not None and vis[child] == 0:
                continue
            if draw_edges:
                cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                        -1)
                cv2.circle(image, (point[0], point[1]), radius - 1,
                        colors[jcolors[child]], -1)
            else:
                # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
                cv2.circle(image, (point[0], point[1]), radius - 1,
                        colors[jcolors[child]], 1)
                # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
            pa_id = parents[child]
            if draw_edges and pa_id >= 0:
                if vis is not None and vis[pa_id] == 0:
                    continue
                point_pa = joints[:, pa_id]
                cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                        colors[jcolors[pa_id]], -1)
                if child not in ecolors.keys():
                    print('bad')
                    import ipdb
                    ipdb.set_trace()
                cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                        colors[ecolors[child]], radius - 2)

        # Convert back in original dtype
        if input_is_float:
            if max_val <= 1.:
                image = image.astype(np.float32) / 255.
            else:
                image = image.astype(np.float32)

        return image
    