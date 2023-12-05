# Youtubehand

### Introduction

---

⚠Unofficial⚠ PyTorch implementation of [Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild](https://arxiv.org/abs/2004.01946), CVPR 2020

![2154_img](https://raw.githubusercontent.com/EAST-J/pictures/main/img/2154_img.jpg)

### Install

---

My codebase is developed based on Ubuntu 18.06 Python 3.7.13 CUDA 11.4.

- Install pytorch and torchvision according to your CUDA and python version.

- Requirements

  ```
  pip install -r requirements.txt
  ```

- Install [MPI-IS Mesh](https://github.com/MPI-IS/mesh)

- You should accept [MANO LICENCE](https://mano.is.tue.mpg.de/license.html). Download the [MANO model](https://mano.is.tue.mpg.de/) and [another files](https://github.com/SeanChenxy/HandMesh/tree/main/template).

The resulting data structure should follow the hierarchy as below.

```
${REP_DIR}
|--conv
|--data
	|--freihand
|--datasets
|--images
|--out
	|checkpoints
	|board
	|demo
	|eval
|template
|utils
|...
|...
```

### Trained Model Download

---

- The pre-trained HRNet can be downloaded according to [METRO](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md)

- Download the trained model from [GoogleDrive](https://drive.google.com/drive/folders/1vw1Rfjdfc4UxWjRR3eFbx2lMjdseN8mh?usp=sharing) (To be updated)

- Here I report my re-produced results on FreiHAND

| Methods    | Backbone  | PA-MPJPE | PA-MPVPE | #Params |
| ---------- | --------- | -------- | -------- | ------- |
| Origin     | ResNet50  | 8.4      | 8.6      | -       |
| Reproduced | ResNet18  | 8.6      | 8.7      | 36M     |
| Reproduced | ResNet50  | 7.7      | 7.7      | 419M    |
| Reproduced | HRNet-W64 | 7.2      | 7.4      | 519M    |

### Demo

---

- Put the input images in the `images` folder 
- Run

```
python main.py --split demo --resume --exp_name $exp_name under the out folder e.g. global-resnet18$
```

### Train

---

- Follow [METRO](https://github.com/microsoft/MeshTransformer) to download FreiHAND dataset.
- Run

```
# resnet18
python main.py --split train --batch_size 64 --epochs 38 --decay_step 30 --backbone resnet18 --out_channels 64 128 256 512 --exp_name global-resnet18
```

### Evaluation

---

- Run

```
python main.py --split eval --exp_name $exp_name under the out folder e.g. global-resnet18$
```

### Acknowledgement

---

The implementation modifies codes or draws inspiration from:

- [HandMesh](https://github.com/SeanChenxy/HandMesh)
- [METRO](https://github.com/microsoft/MeshTransformer)
- [manopth](https://github.com/hassony2/manopth)
