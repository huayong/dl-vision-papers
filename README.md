#### Contents  
- [Camera relocalization](#camera-relocalization)
	- [VLocNet](#vLocnet)
	- [MapNet](#mapnet-githubproject-page)
	- [NNnet](#nnnet-github)
	- [Geometric loss PoseNet](#geometric-loss-posenet)
	- [Hourglass Pose](#hourglass-pose)
	- [VidLoc](#vidloc)
	- [LSTM Pose](#lstm-pose)
	- [BranchNet](#branchnet)
	- [Bayesian PoseNet](#bayesian-posenet)
	- [PoseNet](#posenet-githubproject-page)
- [VO](#vo)
	- [MagicVO](#magicvo)
	- [UnDeepVO](#undeepvo-githubproject-page)
	- [DeepVO](#deepvo-project-page)
- [VIO](#vio)
	- [VINet](#vinet-github)
- [SFM](#sfm)
	- [SfMLearner](#sfmlearner-githubproject-page)
	- [SfM-Net](#sfm-net-github)
	- [DeMoN](#demon-github)
- [MVS](#mvs)
	- [MVSNet](#mvsnet-github)
	- [SurfaceNet](#surfacenet-github)
- [SLAM](#slam)
- [Semantic SLAM](#semantic-slam)
	- [CNN-SLAM](#cnn-slam)
- [Interest Point Detection and Description](#interest-point-detection-and-description)  
	- [SuperPoint](#superpoint-github)
	- [GeoDesc](#geodesc-github)
	- [Quad-networks](#quad-networks)
	- [UCN](#ucn)
	- [LIFT](#lift)
	- [DeepDesc](#deepdesc)
	- [TILDE](#tilde)
	- [*Evaluation*](#evaluation)
- [3D Net](#3d-net)   
  - [PointSift](#pointsift-github) 
  - [PointCNN](#pointcnn-github) 
  - [PointNet++](#pointnet-githubproject-page)  
  - [PointNet](#pointnet-githubproject-page-1) 

------

------

#### Camera Relocalization

##### VLocNet++
[VLocNet++: Deep Multitask Learning for Semantic Visual Localization and
Odometry](https://arxiv.org/abs/1804.08366) [2018 RAL]

##### VLocNet
[Deep Auxiliary Learning for Visual Localization and Odometry](https://arxiv.org/abs/1803.03642) [2018 ICRA]

##### MapNet &nbsp;[[github](https://github.com/NVlabs/geomapnet)]&nbsp;[[project page](https://research.nvidia.com/publication/2018-06_Geometry-Aware-Learning-of)]
[ Geometry-Aware Learning of Maps for Camera Localization](https://arxiv.org/abs/1712.03342) [2018 CVPR]

##### NNnet &nbsp;[[github](https://github.com/AaltoVision/camera-relocalisation)]
[Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network](https://arxiv.org/abs/1707.09733) [2017 ICCV]

##### Geometric loss PoseNet 
[Geometric loss functions for camera pose regression with deep learning](https://arxiv.org/abs/1704.00390) [2017 CVPR]

##### Hourglass Pose
[Image-based Localization using Hourglass Networks](https://arxiv.org/abs/1703.07971) [2017 ICCV]

##### VidLoc
[VidLoc: A Deep Spatio-Temporal Model for 6-DoF Video-Clip Relocalization](https://arxiv.org/abs/1702.06521) [2017 CVPR]

##### LSTM Pose
[Image-based localization using LSTMs for structured feature correlation](https://arxiv.org/abs/1611.07890) [2017 ICCV]

##### BranchNet
[Delving Deeper into Convolutional Neural Networks for Camera Relocalization](http://www.xlhu.cn/papers/Wu17-icra.pdf) [2017 ICRA]

##### Bayesian PoseNet
[Modelling Uncertainty in Deep Learning for Camera Relocalization][https://arxiv.org/abs/1509.05909] [2016 ICRA]

##### PoseNet &nbsp;[[github](https://github.com/alexgkendall/caffe-posenet)]&nbsp;[[project page](http://mi.eng.cam.ac.uk/projects/relocalisation/)]
[PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://arxiv.org/abs/1505.07427) [2015 ICCV]

------

#### VO

##### MagicVO
[MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional
Recurrent Convolutional Neural Network](https://arxiv.org/abs/1811.10964) [2018 arXiv]

##### UnDeepVO &nbsp;[[github](https://github.com/drmaj/UnDeepVO)]&nbsp;[[project page](http://senwang.gitlab.io/UnDeepVO/)]
[UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning](https://arxiv.org/abs/1709.06841) [2018 arXiv]

##### DeepVO &nbsp;[[project page](http://senwang.gitlab.io/DeepVO/)] 
[DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429) [2017 ICRA]

------

#### VIO

##### VINet &nbsp;[[github](https://github.com/HTLife/VINet)]
[VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem](https://arxiv.org/abs/1701.08376) [2017 AAAI]

------

#### SFM

##### SfMLearner &nbsp;[[github](https://github.com/tinghuiz/SfMLearner)]&nbsp;[[project page](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)]
[Unsupervised Learning of Depth and Ego-Motion from Video](https://arxiv.org/abs/1704.07813) [2017 CVPR]

#####SfM-Net &nbsp;[[github](https://github.com/waxz/sfm_net)]
[SfM-Net: Learning of Structure and Motion from Video](https://arxiv.org/abs/1704.07804) [2017 arXiv]

##### DeMoN &nbsp;[[github](https://github.com/lmb-freiburg/demon)]
[DeMoN: Depth and Motion Network for Learning Monocular Stereo](https://arxiv.org/abs/1612.02401) [2017 CVPR]

------

#### MVS

##### MVSNet &nbsp;[[github](https://github.com/YoYo000/MVSNet)]
[MVSNet: Depth Inference for Unstructured Multi-view Stereo](https://arxiv.org/abs/1804.02505) [2018 ECCV]

##### SurfaceNet &nbsp;[[github](https://github.com/mjiUST/SurfaceNet)]
[SurfaceNet: An End-to-end 3D Neural Network for Multiview Stereopsis](https://arxiv.org/abs/1708.01749) [2017 ICCV]

------

#### SLAM

------

#### Semantic SLAM

##### CNN-SLAM
[CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction](https://arxiv.org/abs/1704.03489) [2017 CVPR]

------

#### Depth Estimation

##### monoDepth &nbsp;[[project page](http://visual.cs.ucl.ac.uk/pubs/monoDepth/)]
[Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677) [2017 CVPR]

##### Unsupervised Depth Estimation &nbsp;[[github](https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation)]
[Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue](https://arxiv.org/abs/1603.04992) [2016 ECCV]

------

#### Visual Navigation

##### CMP
[Cognitive Mapping and Planning for Visual Navigation](https://arxiv.org/abs/1702.03920) [2017 CVPR]

##### Target-driven Visual Navigation &nbsp;[[github](https://github.com/zfw1226/icra2017-visual-navigation)]
[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](https://arxiv.org/abs/1609.05143) [2017 ICRA]

------

#### Interest Point Detection and Description

##### SuperPoint &nbsp;[[github](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)]
[SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629) [2017 arXiv]

##### GeoDesc &nbsp;[[github](https://github.com/lzx551402/geodesc)]
[GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://arxiv.org/abs/1807.06294) [2018 ECCV]

##### Quad-networks
[Quad-networks: unsupervised learning to rank for interest point detection](https://arxiv.org/abs/1611.07571) [2017 CVPR]

##### UCN
[Universal Correspondence Network](https://arxiv.org/abs/1606.03558) [2016 NIPS]

##### LIFT
[LIFT: Learned Invariant Feature Transform](https://arxiv.org/abs/1603.09114) [2016 ECCV]

##### DeepDesc
[ Discriminative Learning of Deep Convolutional Feature Point Descriptors](https://icwww.epfl.ch/~trulls/pdf/iccv-2015-deepdesc.pdf) [2015 ICCV]

#####  TILDE
[ TILDE: A Temporally Invariant Learned DEtector](https://arxiv.org/abs/1411.4568) [2015 CVPR]

##### *Evaluation* 
Protocol

[A performance evaluation of local descriptors](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_pami2004.pdf) [2005 PAMI]

Data

[HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors](https://arxiv.org/abs/1704.05939) [2017 CVPR]

------

#### 3D Net

##### PointSift &nbsp;[[github](https://github.com/MVIG-SJTU/pointSIFT)]
[PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/1807.00652) [2018 arXiv]

##### PointCNN &nbsp;[[github](https://github.com/yangyanli/PointCNN)]
[PointCNN: Convolution On X -Transformed Points](https://arxiv.org/abs/1801.07791) [2018 NIPS]

##### PointNet++ &nbsp;[[github](https://github.com/charlesq34/pointnet2)]&nbsp;[[project page](http://stanford.edu/~rqi/pointnet2/)]
[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413) [2017 NIPS]

##### PointNet &nbsp;[[github](https://github.com/charlesq34/pointnet)]&nbsp;[[project page](http://stanford.edu/~rqi/pointnet/)]
[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) [2017 CVPR]


