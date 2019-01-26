#### Contents  
- [SFM](#sfm)
- [MVS](#mvs)
- [SLAM](#slam)
- [Dynamic SLAM](#dynamic-slam)
- [VO](#vo)
- [VIO](#vio)
- [Depth Estimation](#depth-estimation)
- [Flow Estimation](#flow-estimation)

------

------

#### SFM

##### SfMLearner++
[SfMLearner++: Learning Monocular Depth & Ego-Motion using Meaningful Geometric Constraints](https://arxiv.org/abs/1812.08370)&nbsp;[2019 WACV]

##### GeoNet
[GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276)&nbsp;[2018 CVPR]&nbsp;[code: [tensorflow](https://github.com/yzcjtr/GeoNet)]

##### BA-Net
[BA-Net: Dense Bundle Adjustment Network](https://arxiv.org/abs/1806.04807)&nbsp;[2018 arXiv]

##### SfMLearner
[Unsupervised Learning of Depth and Ego-Motion from Video](https://arxiv.org/abs/1704.07813)&nbsp;[2017 CVPR]&nbsp;[code: [tensorflow](https://github.com/tinghuiz/SfMLearner)]&nbsp;[[project page](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)]

> 输入一小段序列帧，估计当前帧的depth以及相对于相邻帧的relative pose，而且重点是不需要训练数据，非监督学习。
> 

##### SfM-Net

[SfM-Net: Learning of Structure and Motion from Video](https://arxiv.org/abs/1704.07804)&nbsp;[2017 arXiv]

##### DeMoN
[DeMoN: Depth and Motion Network for Learning Monocular Stereo](https://arxiv.org/abs/1612.02401)&nbsp;[2017 CVPR]&nbsp;[code: [tensorflow](https://github.com/lmb-freiburg/demon)]

------

#### MVS

##### MVSNet
[MVSNet: Depth Inference for Unstructured Multi-view Stereo](https://arxiv.org/abs/1804.02505)&nbsp;[2018 ECCV]&nbsp;[code: [tensorflow](https://github.com/YoYo000/MVSNet)]

##### SurfaceNet
[SurfaceNet: An End-to-end 3D Neural Network for Multiview Stereopsis](https://arxiv.org/abs/1708.01749)&nbsp;[2017 ICCV]&nbsp;[code: [theano](https://github.com/mjiUST/SurfaceNet)]

------

#### SLAM

##### Fusion++
[Fusion++: Volumetric Object-Level SLAM](https://arxiv.org/abs/1808.08378)&nbsp;[2017 arXiv]

##### Deep SLAM
[Toward Geometric Deep SLAM](https://arxiv.org/abs/1707.07410)&nbsp;[2017 arXiv]

##### CNN-SLAM
[CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction](https://arxiv.org/abs/1704.03489)&nbsp;[2017 CVPR]

##### PAD 
[Probabilistic Data Association for Semantic SLAM](https://www.cis.upenn.edu/~kostas/mypub.dir/bowman17icra.pdf)&nbsp;[2017 ICRA]

##### Semi-Dense 3D Semantic Mapping
[Semi-Dense 3D Semantic Mapping from Monocular SLAM](https://arxiv.org/abs/1611.04144)&nbsp;[2016 arXiv]

##### SemanticFusion
[SemanticFusion: Dense 3D Semantic Mapping with Convolutional Neural Networks](https://arxiv.org/abs/1609.05130)&nbsp;[2017 ICRA]

------

#### Dynamic SLAM

深度学习可以识别场景的各个物体，现在利用深度学习来求解动态场景SLAM问题的工作也越来越多了；

##### MID-Fusion
[MID-Fusion: Octree-based Object-Level Multi-Instance Dynamic SLAM](https://arxiv.org/abs/1812.07976)&nbsp;[2018 arXiv]

##### DS-SLAM
[DS-SLAM: A Semantic Visual SLAM towards Dynamic Environments](https://arxiv.org/abs/1809.08379)&nbsp;[2018 IROS]

##### DynaSLAM
[DynaSLAM: Tracking, Mapping and Inpainting in Dynamic Environments](https://arxiv.org/abs/1806.05620)&nbsp;[2018 RAL]&nbsp;[code: [tensorflow](https://github.com/BertaBescos/DynaSLAM)]

------

#### VO

##### RegNet
[RegNet: Learning the Optimization of Direct Image-to-Image Pose Registration](https://arxiv.org/abs/1812.10212)&nbsp;[2018 arXiv]

##### SIVO
[Self-Improving Visual Odometry](https://arxiv.org/abs/1812.03245)&nbsp;[2018 arXiv]

##### MagicVO
[MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional Recurrent Convolutional Neural Network](https://arxiv.org/abs/1811.10964)&nbsp;[2018 arXiv]

> 1. 这里使用Bi-LSTM，不仅利用前面帧的信息，同时还利用的后面帧的信息（整体和[DeepVO](#deepvo)有点类似）；
> 2. 这里使用Bi-LSTM求解VO，虽然Bi-LSTM能利用前后帧信息，但是在实际使用过程，后面的帧图像无法预先获得；

##### VSO
[VSO: Visual Semantic Odometry](https://demuc.de/papers/lianos2018vso.pdf)&nbsp;[2018 ECCV]

##### LS-VO
[LS-VO: Learning Dense Optical Subspace for Robust Visual Odometry Estimation](https://arxiv.org/abs/1709.06019)&nbsp;[2018 ICRA]&nbsp;[code: [keras](https://github.com/isarlab-department-engineering/LSVO)]

> 1. 在光流结果上利用CNN压缩特征，利用该特征再去求取relative pose；

##### UnDeepVO 
[UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning](https://arxiv.org/abs/1709.06841)&nbsp;[2018 arXiv]&nbsp;[[project page](http://senwang.gitlab.io/UnDeepVO/)]

> 1. 利用已知baseline的双目图像去非监督学习depth(这个也是非监督学习depth的经典做法)；
> 2. 利用上面求解的depth约束前后帧图像photometric一致性以及前后帧3d几何关系一致性，这样loss确定，可以非监督去求解relative pose；

##### DeepVO 

[DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429)&nbsp;[2017 ICRA]&nbsp;[[project page](http://senwang.gitlab.io/DeepVO/)] 

> 1. 前后帧拼接企图特征；
> 2. 特征利用LSTM向后传递，使用两次LSTM；

------

#### VIO

##### VINet
[VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem](https://arxiv.org/abs/1701.08376)&nbsp;[2017 AAAI]

------

#### Depth Estimation

##### Depth VO Feat
[Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction](https://arxiv.org/abs/1803.03893)&nbsp;[2018 CVPR]&nbsp;[code: [caffe](https://github.com/Huangying-Zhan/Depth-VO-Feat)]

##### Vid2Depth
[Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522)&nbsp;[2018 CVPR]&nbsp;[code: [tensorflow](https://github.com/tensorflow/models/tree/master/research/vid2depth)]&nbsp;[[project page](https://sites.google.com/view/vid2depth)]

##### monoDepth
[Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677)&nbsp;[2017 CVPR]&nbsp;[code: [tensorflow](https://github.com/mrharicot/monodepth)]&nbsp;[[project page](http://visual.cs.ucl.ac.uk/pubs/monoDepth/)]

##### Unsupervised Depth Estimation
[Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue](https://arxiv.org/abs/1603.04992)&nbsp;[2016 ECCV]&nbsp;[code: [caffe](https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation)]

------

#### Flow Estimation

##### FlowNet2.0
[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925)&nbsp;[2017 CVPR]&nbsp;[code [caffe](https://github.com/lmb-freiburg/flownet2)]

##### DispNet
[A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation](https://arxiv.org/pdf/1512.02134)&nbsp;[2016 CVPR]&nbsp;[code [caffe](https://lmb.informatik.uni-freiburg.de/resources/binaries/dispflownet/dispflownet-release-1.2.tar.gz)]

##### FlowNet
[FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)&nbsp;[2015 ICCV]&nbsp;[code: [caffe](https://lmb.informatik.uni-freiburg.de/resources/binaries/dispflownet/dispflownet-release-1.2.tar.gz)]&nbsp;[[project page](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/)]