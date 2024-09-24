# SatNeRF Docker Image Tutorial

이 튜토리얼은 SatNeRF Docker 이미지를 사용하여 위성 신경 방사장(NeRF) 작업을 수행하는 방법을 단계별로 제공합니다. 여기에는 테스트, 훈련 및 추가 실험이 포함됩니다.

## 목차

1. [Docker 이미지 가져오기](#1-docker-이미지-가져오기)
2. [Docker 이미지 실행하기](#2-docker-이미지-실행하기)
3. [데이터셋 및 사전 학습 모델 위치 설정](#3-데이터셋-및-사전-학습-모델-위치-설정)
4. [표면 모델 생성 테스트](#4-표면-모델-생성-테스트)
5. [새로운 시점 합성 테스트](#5-새로운-시점-합성-테스트)
6. [모델 훈련](#6-모델-훈련)
7. [추가 실험](#7-추가-실험)
   - [7.1 데이터셋 생성](#71-데이터셋-생성)
   - [7.3 Depth Supervision](#73-Depth-Supervision)
   - [7.3 Interpolate over different sun directions](#73-Interpolate-over-different-sun-directions)
   - [7.4 클래식 위성 MVS와의 비교](#74-클래식-위성-mvs와의-비교)
8. [참고 사항](#8-참고-사항)
9. [결론](#9-결론)
10. [원본 소스 및 저작권](#10-원본-소스-및-저작권)

---

## 1. Docker 이미지 가져오기

먼저, Docker Hub에서 SatNeRF Docker 이미지를 가져옵니다:

```bash
docker pull your_dockerhub_username/satnerf-image:latest
```

*`your_dockerhub_username`을 실제 Docker Hub 사용자 이름으로 대체하세요.*

---

## 2. Docker 이미지 실행하기

다음 명령어로 Docker 이미지를 실행합니다:

```bash
docker run -it --gpus all --cpus="4" --rm \
    -v $(pwd)/pretrained_models:/mnt/cdisk/roger/EV2022_satnerf/pretrained_models \
    -v $(pwd)/dataset:/mnt/cdisk/roger/EV2022_satnerf/dataset \
    -v $(pwd)/output:/mnt/cdisk/roger/EV2022_satnerf/out_dsm_path \
    satnerf-image bash
```

**설명:**

- `--gpus all`: 컨테이너 내에서 모든 GPU를 사용하도록 설정합니다.
- `--cpus="4"`: 컨테이너가 4개의 CPU 코어를 사용하도록 제한합니다.
- `--rm`: 컨테이너가 종료될 때 자동으로 제거합니다.
- `-v $(pwd)/pretrained_models:/mnt/cdisk/roger/EV2022_satnerf/pretrained_models`: 로컬의 `pretrained_models` 디렉토리를 컨테이너에 마운트합니다.
- `-v $(pwd)/dataset:/mnt/cdisk/roger/EV2022_satnerf/dataset`: 로컬의 `dataset` 디렉토리를 컨테이너에 마운트합니다.
- `-v $(pwd)/output:/mnt/cdisk/roger/EV2022_satnerf/out_dsm_path`: 로컬의 `output` 디렉토리를 컨테이너에 마운트합니다.

**참고:** 현재 작업 디렉토리에 `pretrained_models`, `dataset`, `output` 디렉토리가 존재하는지 확인하세요.

---

## 3. 데이터셋 및 사전 학습 모델 위치 설정

Docker 컨테이너 내부에서 데이터셋과 사전 학습 모델에 대한 환경 변수를 설정합니다:

```bash
export dataset_dir=/mnt/cdisk/roger/EV2022_satnerf/dataset
export pretrained_models=/mnt/cdisk/roger/EV2022_satnerf/pretrained_models
```

---

## 4. 표면 모델 생성 테스트

Sat-NeRF를 사용하여 표면 모델을 생성하기 위한 코드입니다:

```bash
python3 create_satnerf_dsm.py Sat-NeRF $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_dsm_path/JAX_068 28 \
    $pretrained_models/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 \
    $dataset_dir/DFC2019/Track3-Truth
```

---

## 5. 새로운 시점 합성 테스트

Sat-NeRF를 사용하여 새로운 시점 합성을 테스트하기 위한 코드입니다:

```bash
python3 eval_satnerf.py Sat-NeRF $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_eval_path/JAX_068 28 val \
    $pretrained_models/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 \
    $dataset_dir/DFC2019/Track3-Truth
```

---

## 6. 모델 훈련

Sat-NeRF 모델을 처음부터 훈련하려면 다음 명령어를 사용합니다:

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

**참고:** 경로가 다를 경우 데이터셋과 출력 디렉토리에 맞게 경로를 조정해야 합니다.

---

## 7. 추가 실험

### 7.1 데이터셋 생성

데이터셋을 생성하려면 다음을 실행합니다:

```bash
python3 create_satellite_dataset.py JAX_068 \
    $dataset_dir/DFC2019 \
    /mnt/cdisk/roger/EV2022_satnerf/out_dataset_path/JAX_068
```

---

### 7.3 Depth Supervision

Depth Supervision의 효과 비교를 시행하려면 다음 명령어를 실행합니다:

```bash
python3 study_depth_supervision.py Sat-NeRF+DS $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_DS_study_path/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops \
    $dataset_dir/DFC2019/Track3-Truth
```

---

### 7.3 Interpolate over different sun directions

다양한 태양 방향에 대해 보간하려면 다음을 실행합니다:

```bash
python3 study_solar_interpolation.py Sat-NeRF $pretrained_models/JAX_068 \
    /mnt/cdisk/roger/EV2022_satnerf/out_solar_study_path/JAX_068 28 \
    $pretrained_models/JAX_068 \
    $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 \
    $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 \
    $dataset_dir/DFC2019/Track3-Truth
```

---

### 7.4 클래식 위성 MVS와의 비교

클래식 위성 MVS 파이프라인과 비교하려면 다음을 실행하세요:

```bash
python3 eval_s2p.py JAX_068 \
    /mnt/cdisk/roger/Datasets/SatNeRF/root_dir/fullaoi_rpcs_ba_v1/JAX_068 \
    /mnt/cdisk/roger/Datasets/DFC2019 \
    /mnt/cdisk/roger/nerf_output-crops3/results \
    --n_pairs 10
```

---

## 8. 참고 사항

- **실행 시 주의사항:** Docker를 실행할 때 CPU 코어를 4개 이상 할당하는 것이 좋습니다.

---

## 9. 결론

이 튜토리얼은 SatNeRF Docker 이미지를 사용하여 다양한 작업을 수행하기 위한 필요한 명령어와 단계를 제공합니다. 이 단계를 따라 SatNeRF 모델을 성공적으로 실행하고 필요한 실험을 수행할 수 있습니다.

---

## 10. 원본 소스 및 저작권

이 프로젝트에서 사용된 Dockerfile 및 스크립트는 Centre Borelli, MINES ParisTech, PSL University에서 개발한 [SatNeRF 리포지토리](https://github.com/centreborelli/satnerf.git)를 기반으로 합니다.

원본 소스 코드와 SatNeRF 프로젝트에 대한 자세한 내용은 공식 리포지토리에서 확인할 수 있습니다: [https://github.com/centreborelli/satnerf.git](https://github.com/centreborelli/satnerf.git).

**저작권 공지**

SatNeRF 코드 및 관련 리소스는 원저자의 [SatNeRF GitHub 리포지토리](https://github.com/centreborelli/satnerf.git)에 명시된 해당 조건에 따라 라이선스가 부여됩니다. 원저자가 제공한 라이선스 지침을 반드시 따라야 합니다.

이 프로젝트는 원본 SatNeRF 코드에 대한 소유권을 주장하지 않으며, 원저자가 제공한 조건에 따라 배포됩니다. 자세한 내용은 원본 [LICENSE](https://github.com/centreborelli/satnerf/blob/main/LICENSE) 파일을 참조하세요.

---

*필요에 따라 이 README를 사용자 지정하거나 추가 세부 정보를 포함할 수 있습니다.*
