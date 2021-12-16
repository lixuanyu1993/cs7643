# CS7643 Final Project
## Feng Dai, Bo Lin, Boyan Lu, Xuanyu Li

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This is a project for the course cs7643 final project for AI-Habitat
It has two parts:
- RL
- VO
## Steps
1. Generate training data
2. Training VO
3. Training RL
4. Validate Result
## Prerequest:
Linux with Cuda

### Install Habitat
The repo is tested under the following commits of habitat-lab and habitat-sim.
habitat-lab == d0db1b55be57abbacc5563dca2ca14654c545552
habitat-sim == 020041d75eaf3c70378a9ed0774b5c67b9d3ce99

## Generate training data

We used the dataset from the gibson. Link:
http://gibsonenv.stanford.edu/database/
https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets

```
cd VO_side
python habitat_model/vo/dataset/generate_datasets.py 
--config_f ./configs/point_nav_habitat_challenge_2020.yaml 
--train_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/train/content 
--val_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/train/content/
--save_dir ./dataset/vo_dataset/ 
--data_version v2 
--vis_size_w 341 
--vis_size_h 192 
--obs_transform none 
--act_type -1 
--rnd_p 1.0 
--N_list 10000000
--name_list train
```
### VO data Structure
```
-----VO_side
| ----dataset
|  | ----Gibson
|  |  |----gibson
|  |  |  | ---- *.glb
|  |  |  | ---- *.navmesh
|  | ----habitat_datasets
|  |  |----pointnav
|  |  |  | ---- glibson
|  |  |  |  | ---- v2
|  |  |  |  |  | ---- train
|  |  |  |  |  | ---- val
|  | ----vo_dataset
|  |  | ----train_10000000.h5

### RL data Structure
-----RL_side
| ----data
|  | ----datasets
|  |  |----pointnav
|  |  |  | ---- glibson
|  |  |  |  | ---- v1
|  |  |  |  |  | ---- train
|  |  |  |  |  | ---- val
|  | ----scene_datasets
|  |  | ---- glibson
|  |  |  | ---- *.glb
|  |  |  | ---- *.navmesh
```

## Training VO
```
cd VO_side
python run.py --task-type vo --run-type train
```

## Training RL
```
cd RL_side
python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_debug.yaml --run-type train
```
## Val
```
cd VO_side
python ./run.py --taks-type rl --run-type eval
```
