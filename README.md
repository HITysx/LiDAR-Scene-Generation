# Enhancing Realism in LiDAR Scene Generation with CSPA-DFN and Linear Cross-Attention via Diffusion Transformer Model
This repository contains the code implementation of the paper "Enhancing Realism in LiDAR Scene Generation with CSPA-DFN and Linear Cross-Attention via Diffusion Transformer Model"

<div align="center">
  <img src="https://github.com/HITysx/LiDAR-Scene-Generation/blob/main/assets/lidar%20diffusion%20transformer.pdf">
</div>

## Abstract
This paper proposes a novel LiDAR Diffusion Transformer Model that integrates Channel-Spatial Parallel Attention and Dilation Fusion Network (CSPA-DFN) with a linear cross-attention post-processing module to refine the generated LiDAR scene samples. The model has been evaluated on the unconditional generation task using the KITTI-360 and nuScenes datasets. Furthermore, by incorporating semantic labels and camera views into the latent space, in addition to enhancing the model's semantic understanding capability for LiDAR scenes, the method also demonstrates additional performance improvements compared to previous works in terms of LiDAR scene's visual quality.

## Data Preparation
KITTI-360
1. Download KITTI-360 from [http://www.cvlibs.net/datasets/kitti-360/](http://www.cvlibs.net/datasets/kitti-360/)
   (For unconditional generation task, only the 3D LiDAR readings are required. For Camera-to-LiDAR task, both the 3D LiDAR readings and 2D image readings are required.)
2. Set the KITTI360 dataset path in Class KITTI360Train(KITTI360Val) of 'DiT/data/kitti.py'.

Semantic-Map-to-LiDAR
1. Download SemanticKITTI from [http://semantic-kitti.org/index.html](http://semantic-kitti.org/index.html)
   (Only 00-10 sequences are required for this task. We use 08 sequence for validation sets and other sequences for training sets)
2. Set the SemanticKITTI dataset path in Class SemanticKITTITrain(SemanticKITTIVal) of 'DiT/data/kitti.py'.

nuScenes
1. Download nuScenes from [https://www.nuscenes.org/](https://www.nuscenes.org/)
2. Set the nuScenes dataset path in Class NuScenesTrain(NuScenesVal) of 'DiT/data/nuScenes.py'

## Requirements
Create a new conda environment named lidar_scene_generation
```
sh init/create_env.sh
conda activate lidar_scene_generation
```

## Training

```
# train an autoencoder
python main.py -b configs/autoencoder/kitti/autoencoder_c2_p4.yaml -t --gpus 0,1,2,3

# train an diffusion transformer model
python main.py -b configs/lidar_scene_generation/kitti/uncond_c2_p4.yaml -t --gpus 0,1,2,3
```

To resume your training from an existing checkpoint file, use the flag `-r`:

```
python main.py -b path/to/your/config.yaml -t --gpus 0, -r path/to/your/ckpt/file
```

## Sample

Unconditional generation

To run sampling on pretrained models (and to evaluate your results with flag "--eval"), firstly download our provided pretrained autoencoders to directory `./models/first_stage_models/kitti/[model_name]` and pretrained models to directory `./models/lidar_scene_generation/kitti/[model_name]`:

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py -d kitti -r models/lidar_scene_generation/kitti/uncond/model.ckpt -n 2000 [--eval]
```

Semantic-Map-to-LiDAR

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample_cond.py -r models/lidar_scene_generation/kitti/sem2lidar/model.ckpt -d kitti [--eval]
```

Camera-to-LiDAR

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample_cond.py -r models/lidar_scene_generation/kitti/cam2lidar/model.ckpt -d kitti [--eval]
```

## Evaluation

For the details about setup and usage of evaluation toolbox, please refer to the [Evaluation toolbox README.md].

To evaluate the model through the given .pcd files:

```
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py -d kitti -f models/lidar_scene_generation/kitti/[method]/samples.pcd --eval
```

## Acknowledgement

- Our codebase for the diffusion models builds heavily on [LiDAR Diffusion](https://github.com/hancyran/LiDAR-Diffusion).












