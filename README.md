# HSTFormer: Hierarchical Spatial-Temporal Transformers for 3D Human Pose Estimation

Unofficial implementation of paper([HSTFormer: Hierarchical Spatial-Temporal Transformers for 3D Human Pose Estimation](https://arxiv.org/abs/2301.07322)).

해당 논문의 Hierarchical STE, JTTE, BTTE의 Module을 구현하였음, 하지만, PTTE같은 경우, Attention 연산량의 한계로 MixSTE의 ST Module로 대체함

<p align="center"> <img src="./assets/SittingDown_s1.gif" width="80%"> </p> 
<p align="center"> Visualization of our method and ground truth on Human3.6M </p>

## Environment

다음 환경에서 진행하였음

* Ubuntu 20.04
* Python 3.8.18
* PyTorch 1.13.0+cu116
* CUDA 11.7


## Dataset

The Human3.6M datase은 [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)에서 제공한 2D Skeleton Squence를 사용하였음
./data 디렉토리 생성 후, 해당 디렉토리 안에 data_2d_h36m_cpn_ft_h36m_dbb.npz, data_2d_h36m_gt.npz, data_3d_h36m.np를 세팅

# Evaluation

* [ ] Download the checkpoints [Google Drive](https://drive.google.com/file/d/1cvEudUAo5F_QYyhLKAwEsUiHiMBa8TRH/view?usp=sharing);

checkpoint model를 기반으로 평가

> python run.py -c <checkpoint_path> --evaluate <checkpoint_file> --gpu 0,1

# Training from scratch

2개 GPUs 사용하여 Training 243 frames:

>  python run.py -f 243 -s 243 -l log/run -c checkpoint -gpu 0,1


# Visulization

Please refer to the https://github.com/facebookresearch/VideoPose3D#visualization.

## Acknowledgement

Thanks for the baselines, we construct the code based on them:

* VideoPose3D
* MixSTE
