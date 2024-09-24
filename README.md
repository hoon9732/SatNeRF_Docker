# SatNeRF Docker Image Tutorial

This tutorial provides step-by-step instructions on how to use the SatNeRF Docker image for satellite neural radiance field tasks, including testing, training, and additional experiments.

## Table of Contents

1. [Pull the Docker Image](#1-pull-the-docker-image)
2. [Run the Docker Image](#2-run-the-docker-image)
3. [Set Dataset and Pretrained Models Location](#3-set-dataset-and-pretrained-models-location)
4. [Testing Generation of Surface Model](#4-testing-generation-of-surface-model)
5. [Testing Novel View Synthesis](#5-testing-novel-view-synthesis)
6. [Training the Model](#6-training-the-model)
7. [Additional Experiments](#7-additional-experiments)
   - [7.1 Dataset Creation](#71-dataset-creation)
   - [7.3 Depth Supervision](#73-depth-supervision)
   - [7.3 Interpolate Over Different Sun Directions](#73-interpolate-over-different-sun-directions)
   - [7.4 Comparison to Classic Satellite MVS](#74-comparison-to-classic-satellite-mvs)
8. [Notes](#8-notes)
   
---

## 1. Pull the Docker Image

First, pull the SatNeRF Docker image from Docker Hub:

```bash
docker pull your_dockerhub_username/satnerf-image:latest
```

*Replace `your_dockerhub_username` with your actual Docker Hub username.*

---

## 2. Run the Docker Image

Run the Docker image with the following command:

```bash
docker run -it --gpus all --cpus="4" --rm \
    -v $(pwd)/pretrained_models:/mnt/cdisk/roger/EV2022_satnerf/pretrained_models \
    -v $(pwd)/dataset:/mnt/cdisk/roger/EV2022_satnerf/dataset \
    -v $(pwd)/output:/mnt/cdisk/roger/EV2022_satnerf/out_dsm_path \
    satnerf-image bash
```

**Explanation:**

- `--gpus all`: Enables all GPUs for use within the container.
- `--cpus="4"`: Limits the container to use 4 CPU cores.
- `--rm`: Automatically removes the container when it exits.
- `-v $(pwd)/pretrained_models:/mnt/cdisk/roger/EV2022_satnerf/pretrained_models`: Mounts your local `pretrained_models` directory to the container.
- `-v $(pwd)/dataset:/mnt/cdisk/roger/EV2022_satnerf/dataset`: Mounts your local `dataset` directory to the container.
- `-v $(pwd)/output:/mnt/cdisk/roger/EV2022_satnerf/out_dsm_path`: Mounts your local `output` directory to the container.

**Note:** Ensure that your local directories `pretrained_models`, `dataset`, and `output` exist in your current working directory.

---

## 3. Set Dataset and Pretrained Models Location

Inside the Docker container, set the environment variables for the dataset and pretrained models:

```bash
export dataset_dir=/mnt/cdisk/roger/EV2022_satnerf/dataset
export pretrained_models=/mnt/cdisk/roger/EV2022_satnerf/pretrained_models
```

---

## 4. Testing Generation of Surface Model

To generate a surface model with Sat-NeRF, run the following command inside the Docker container:

```bash
python3 create_satnerf_dsm.py Sat-NeRF $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_dsm_path/JAX_068 28 \
    $pretrained_models/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 \
    $dataset_dir/DFC2019/Track3-Truth
```

---

## 5. Testing Novel View Synthesis

To test novel view synthesis with Sat-NeRF, run:

```bash
python3 eval_satnerf.py Sat-NeRF $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_eval_path/JAX_068 28 val \
    $pretrained_models/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 \
    $dataset_dir/DFC2019/Track3-Truth
```

---

## 6. Training the Model

To train the Sat-NeRF model from scratch, use the following command:

```bash
python3 main.py --model sat-nerf \
    --exp_name JAX_068_ds1_sat-nerf \
    --root_dir /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/JAX_068 \
    --img_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-RGB-crops/JAX_068 \
    --cache_dir /mnt/cdisk/roger/Datasets/SatNeRF/cache_dir/crops_rpcs_ba_v2/JAX_068_ds1 \
    --gt_dir /mnt/cdisk/roger/Datasets/DFC2019/Track3-Truth \
    --logs_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/logs \
    --ckpts_dir /mnt/cdisk/roger/Datasets/SatNeRF_output/ckpts
```

**Note:** Adjust the paths to match your dataset and output directories if they differ.

---

## 7. Additional Experiments

### 7.1 Dataset Creation

To create the dataset, run:

```bash
python3 create_satellite_dataset.py JAX_068 \
    $dataset_dir/DFC2019 \
    /mnt/cdisk/roger/EV2022_satnerf/out_dataset_path/JAX_068
```

---

### 7.3 Depth Supervision

To study the effect of depth supervision, execute:

```bash
python3 study_depth_supervision.py Sat-NeRF+DS $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_DS_study_path/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops \
    $dataset_dir/DFC2019/Track3-Truth
```

---

### 7.3 Interpolate Over Different Sun Directions

To interpolate over different sun directions, run:

```bash
python3 study_solar_interpolation.py Sat-NeRF $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_solar_study_path/JAX_068 28 \
    $pretrained_models/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 \
    $dataset_dir/DFC2019/Track3-Truth
```

---

### 7.4 Comparison to Classic Satellite MVS

To compare to a classic satellite MVS pipeline, execute:

```bash
python3 eval_s2p.py JAX_068 \
    /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/fullaoi_rpcs_ba_v1/JAX_068 \
    /mnt/cdisk/roger/Datasets/DFC2019 \
    /mnt/cdisk/roger/nerf_output-crops3/results \
    --n_pairs 10
```

---

## 8. Notes

- **Runtime:** Docker에 실행시 cpu 코어를 4개 이상 할당해주는 것이 좋다.
---


*Feel free to customize this README according to your specific needs or add additional details as required.*
