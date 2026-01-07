<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS529.Q11 - RESEARCH ISSUES AND APPLICATIONS IN COMPUTER SCIENCE</b></h1>

## TABLE OF CONTENTS
* [Course Introduction](#course-introduction)
* [Instructor](#instructor)
* [Students](#students)
* [Project](#project)
* [Contents](#contents)
* [Environment Setting](#environment-setting)
* [Dataset](#dataset)
  - [Download](#download)
  - [Preparation](#preparation)
* [Pretrained Model](#pretrained-model)
* [Evaluation](#evaluation)
* [Training](#training)
* [Demo](#demo)
* [Web Demo](#web-demo)
* [License](#license)

## COURSE INTRODUCTION
<a name="course-introduction"></a>
* **Course Name**: Research Issues and Applications in Computer Science
* **Course Code**: CS529
* **Class**: CS529.Q11
* **Start Date**: September 8, 2025
* **Academic Year**: 2025 - 2026

## INSTRUCTOR
<a name="instructor"></a>
* ThS. **Đỗ Văn Tiến** - *tiendv@uit.edu.vn*

## STUDENTS
<a name="students"></a>
| Student ID | Name              | Github                                               | Email                   |
|:----------:|:-----------------:|:----------------------------------------------------:|:-----------------------:|
| 22521587   | Trương Phúc Trường | [Truong99zvc](https://github.com/Truong99zvc/)       | 22521587@gm.uit.edu.vn  |
| 22521571   | Võ Đình Trung     | [votrung654](https://github.com/votrung654/)         | 22521571@gm.uit.edu.vn  |

## PROJECT
<a name="project"></a>
**Project Title**: EVALUATION, REPRODUCTION, AND EXTENSION OF BIDIRECTIONAL MOTION FIELD-GUIDED VIDEO FRAME INTERPOLATION FOR COMPLEX MOTION SCENARIOS

This repository contains the implementation and evaluation of BiM-VFI (Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions), a state-of-the-art video frame interpolation method proposed in CVPR 2025. The project focuses on evaluating, reproducing, and extending the original BiM-VFI framework for handling complex motion scenarios.

## CONTENTS
- [Contents](#contents)
- [Environment Setting](#environment-setting)
- [Dataset](#dataset)
  - [Download](#download)
  - [Preparation](#preparation)
- [Pretrained Model](#pretrained-model)
- [Evaluation](#evaluation)
- [Training](#training)
- [Demo](#demo)
- [Web Demo](#web-demo)
- [License](#license)

## ENVIRONMENT SETTING
To run this project, you need to set up your environment as follows:

```bash
conda create -n bimvfi python=3.11
conda activate bimvfi
pip install basicsr-fixed Ipython torchsummary wandb moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch==2.4.1 torchvision==0.19.1
conda install cupy -c conda-forge
```

**Additional libraries required for Web Demo:**
```bash
pip install flask werkzeug pillow scikit-image
```

## DATASET
### Download
You can download the Vimeo90K dataset used for training and testing from the following link:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)

### Preparation
After downloading the Vimeo90K dataset, organize it according to the project structure. The dataset should be placed in the `data` directory with the appropriate folder structure.

## PRETRAINED MODEL
Pre-trained model can be downloaded from [here](https://drive.google.com/file/d/18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC/view?usp=sharing).

Place the downloaded model file in the `pretrained` directory.

## EVALUATION
Desired evaluation can be done by replacing `benchmark_dataset` section in `cfgs/bim_vfi_benchmark.yaml`.
* `name`: Name of benchmark datasets. The datasets that can be benchmarked are [_vimeo_, _vimeo\_septuplet_, _snu\_film_, _snu\_film\_arb_, _xtest_].
* `args`:
  * `root_path`: Path of each dataset.
  * `split`: Desired splits to evaluate. [_test_, _val_] for _vimeo_ and _vimeo\_septuplet_, [(_easy_), _medium_, _hard_, _extreme_] for _snu\_film_ and _snu\_film\_arb_, and [_single_, _multiple_] for _xtest_.
  * `pyr_lvl`: 3 for vimeo, 5 for snu_film, and 7 for xtest.
* `save_imgs`: `True` if you want to save interpolation results, else `False`. It takes much more time to save images.

Then, run below:
```bash
python main.py --cfg cfgs/bim_vfi_benchmark.yaml
```

## TRAINING
This project implements two models: **BiM-RNet** and **BiM-VFI**. Each model has its own configuration file for training.

### Training BiM-RNet
For single GPU training of BiM-RNet:
```bash
python main.py --cfg cfgs/bim_rnet.yaml
```

For multiple GPU training with GPU number 0, 1, 2, 3:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_rnet.yaml
```

### Training BiM-VFI
For single GPU training of BiM-VFI:
```bash
python main.py --cfg cfgs/bim_vfi.yaml
```

For multiple GPU training with GPU number 0, 1, 2, 3:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_vfi.yaml
```

To run with wandb, fill in wandb.yaml and run with:
```bash
python main.py --cfg cfgs/bim_vfi.yaml -w
```

## DEMO
Custom videos in multiple images or video format can be interpolated as follows.

First, set demo root directory as follow:
  - video1.mp4 
  - video2.mp4
  - video3
    - img0.png
    - img1.png
    - ...
  - ...

Then, replace root_path in `cfgs/bim_vfi_demo.yaml` to desired data root, and run:
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

## WEB DEMO
A web-based demo application is available for interactive video frame interpolation. The web demo allows users to upload images or videos and perform frame interpolation through a user-friendly interface.

### Running the Web Demo

1. Navigate to the web_demo directory:
```bash
cd web_demo
```

2. Ensure all required dependencies are installed (see [Environment Setting](#environment-setting) for the complete list including Flask and related libraries).

3. Make sure the pretrained model is available in the `pretrained` directory (see [Pretrained Model](#pretrained-model)).

4. Run the Flask application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:5000
```

### Web Demo Features
- **Image Interpolation**: Upload two consecutive frames to generate interpolated frames between them
- **Video Interpolation**: Upload a video file to increase its frame rate through interpolation
- **Frame Sequence Interpolation**: Upload multiple frames to create smooth transitions
- **Model Selection**: Choose between different pretrained models
- **Real-time Processing**: View interpolation results with timing statistics

The web demo automatically handles image resizing, padding, and post-processing to ensure optimal results.

## LICENSE
The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr).

