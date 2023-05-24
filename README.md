# Youtubehand

### Introduction

---

⚠Unofficial⚠ PyTorch implementation of [Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild](https://arxiv.org/abs/2004.01946), CVPR 2020

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

- Download the trained model from [GoogleDrive](https://drive.google.com/drive/folders/1vw1Rfjdfc4UxWjRR3eFbx2lMjdseN8mh?usp=sharing)

- Here I report the re-produced results on FreiHAND

| Methods    | Backbone | PA-MPJPE | PA-MPVPE |
| ---------- | -------- | -------- | -------- |
| Origin     | ResNet50 | 8.4      | 8.6      |
| Reproduced | ResNet18 | 9.0      | 9.1      |
| Reproduced | ResNet50 | 7.9      | 8.2      |



### Demo

---

- Put the input images in the `images` folder 
- Run

```
python main.py --split demo --resume $path to trained model$
```

### Train

---

- Follow [METRO](https://github.com/microsoft/MeshTransformer) to download FreiHAND dataset.
- Run

```
python main.py --split train --batch_size 64 --epochs 38 --decay_step 30 --backbone resnet18 --out_channels 64 128 256 512
```

### Evaluation

---

- Run

```
python main.py --split eval --resume $path to trained model$
```

### Acknowledgement

---

The implementation modifies codes or draws inspiration from:

- [HandMesh](https://github.com/SeanChenxy/HandMesh)

- [METRO](https://github.com/microsoft/MeshTransformer)
- [manopth](https://github.com/hassony2/manopth)
