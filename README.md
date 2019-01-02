#### Contents  
- [3D Object Recon](#3d-object-recon)	
- [3D Plane Recon](#3d-plane-recon)
  - [PlaneRCNN](#planercnn)
  - [PlaneRecover](#planerecover)
  - [PlaneNet](#planenet)
- [3D Scene Recon](#3d-scene-recon)
  - [Factored3d](#factored3d)
  - [3DSPR](#3dspr)
- [Apperence Transfer](#apperence-transfer)
  - [ToDayGAN](#todaygAN)
  - [ATLoc](#atloc)
  - [UNIT](#unit)
  - [CycleGAN](#pix2pix)
  - [Pix2Pix](#pix2pix)
- [Camera relocalization](#camera-relocalization)
  - [VLocNet++](#vLocnet)
  - [VLocNet](#vLocnet-1)
  - [MapNet](#mapnet)
  - [NNnet](#nnnet)
  - [Geometric loss PoseNet](#geometric-loss-posenet)
  - [Hourglass Pose](#hourglass-pose)
  - [VidLoc](#vidloc)
  - [LSTM PoseNet](#lstm-posenet)
  - [DSAC](#dsac)
  - [BranchNet](#branchnet)
  - [Bayesian PoseNet](#bayesian-posenet)
  - [PoseNet](#posenet)
- [VO](#vo)
  - [MagicVO](#magicvo)
  - [UnDeepVO](#undeepvo)
  - [DeepVO](#deepvo)
- [VIO](#vio)
  - [VINet](#vinet)
- [SFM](#sfm)
  - [GeoNet](#geonet)
  - [SfMLearner](#sfmlearner)
  - [SfM-Net](#sfm-net)
  - [DeMoN](#demon)
- [MVS](#mvs)
  - [MVSNet](#mvsnet)
  - [SurfaceNet](#surfacenet)
- [SLAM](#slam)
  - [Deep SLAM](#deep-slam)
- [Semantic SLAM](#semantic-slam)
  - [CNN-SLAM](#cnn-slam)
- [Depth Estimation](#depth-estimation)
	- [monoDepth](#monodepth)
	- [Unsupervised Depth Estimation](#unsupervised-depth-estimation)
- [Flow Estimation](#flow-estimation)
	- [FlowNet2.0](#flowNet20)
	- [FlowNet](flownet)
- [Local Feature](#local-feature)  
  - [SuperPoint](#superpoint)
  - [GeoDesc](#geodesc)
  - [Quad-networks](#quad-networks)
  - [UCN](#ucn)
  - [LIFT](#lift)
  - [DeepDesc](#deepdesc)
  - [TILDE](#tilde)
  - [*Evaluation*](#evaluation)
- [3D Net](#3d-net)   
  - [PointSift](#pointsift) 
  - [PointCNN](#pointcnn) 
  - [PointNet++](#pointnet)  
  - [PointNet](#pointnet-1) 

------

------

#### 3D Object Recon

------

#### 3D Plane Recon

##### PlaneRCNN
[PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image](https://arxiv.org/abs/1812.04072)&nbsp;[2018 arXiv]

##### PlaneRecover
[Recovering 3D Planes from a Single Image via Convolutional Neural Networks](https://arxiv.org/pdf/1812.04072)&nbsp;[2018 ECCV]&nbsp;[code: [tensorflow](https://github.com/fuy34/planerecover)]

##### PlaneNet
[PlaneNet: Piece-wise Planar Reconstruction from a Single RGB Image](https://arxiv.org/abs/1804.06278)&nbsp;[2018 CVPR]&nbsp;[code: [tensorflow](https://github.com/art-programmer/PlaneNet)]&nbsp;[[project page](http://art-programmer.github.io/planenet.html)]

------

#### 3D Scene Recon

##### Factored3d
[Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene](https://arxiv.org/abs/1712.01812)&nbsp;[2018 CVPR]&nbsp;[code: [pytorch](https://github.com/shubhtuls/factored3d)]&nbsp;[[project page](https://shubhtuls.github.io/factored3d/)]

##### 3DSPR

[Holistic 3D Scene Parsing and Reconstruction from a Single RGB Image](https://arxiv.org/abs/1808.02201)&nbsp;[2018 ECCV]&nbsp;[code: [python](https://github.com/thusiyuan/holistic_scene_parsing)]&nbsp;[[project page](http://siyuanhuang.com/holistic_parsing/main.html)]

------

#### Apperence Transfer

##### ToDayGAN
[Night-to-Day Image Translation for Retrieval-based Localization](https://arxiv.org/abs/1809.09767)&nbsp;[2018 arXiv]&nbsp;[code: [pytorch](https://github.com/AAnoosheh/ToDayGAN)]

##### ATLoc
[Adversarial Training for Adverse Conditions: Robust Metric Localisation using Appearance Transfer](https://arxiv.org/abs/1803.03341) [2018 CoRR]

##### UNIT
[Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848 )&nbsp;[2017 NIPS]&nbsp;[code: [pytorch](https://github.com/mingyuliutw/UNIT)]

------

#### Image Retrieval
or called Place Recognition and Image Representation and Image Desciptor

##### DLEF
[ Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/pdf/1612.06321.pdf)&nbsp;[2017 ICCV]&nbsp;[code: [tensorflow](https://github.com/tensorflow/models/tree/master/research/delf)]

##### PlaNet
[PlaNet - Photo Geolocation with Convolutional Neural Networks](https://arxiv.org/abs/1602.05314)&nbsp;[2016 ECCV]

##### NetVLAD
[NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247)&nbsp;[2016 CVPR]&nbsp;[code: [matconvnet](https://github.com/Relja/netvlad)]&nbsp;[[project page](https://www.di.ens.fr/willow/research/netvlad/)]

##### DenseVLAD
[24/7 place recognition by view synthesis](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/Torii-CVPR-2015-final.pdf)&nbsp;[2015 CVPR]&nbsp;[code: [c++](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/)]&nbsp;[[project page](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/)]

##### *Benchmark*

------

#### Senmantic Relocalization

##### SIVO
[Visual SLAM with Network Uncertainty Informed Feature Selection](https://arxiv.org/abs/1811.11946)&nbsp;[2018 arXiv]&nbsp;[code: [caffe](https://github.com/navganti/SIVO)]

##### DeLS-3D
[DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map](https://arxiv.org/abs/1805.04949)&nbsp;[2018 CVPR]&nbsp;[code: [only test](https://github.com/pengwangucla/DeLS-3D)]

##### Semantic Visual Localization
[Semantic Visual Localization](https://arxiv.org/abs/1712.05773)&nbsp;[2018 CVPR]&nbsp;[code: [only test](https://github.com/pengwangucla/DeLS-3D)]

##### SSI
[Long-term Visual Localization using Semantically Segmented Images](https://arxiv.org/abs/1801.05269)&nbsp;[2018 ICRA]

##### Semantic Match Consistency
[Semantic Match Consistency for Long-Term Visual Localization](http://people.inf.ethz.ch/sattlert/publications/Toft2018ECCV.pdf)&nbsp;[2018 ECCV]

##### SL
[Long-term 3D Localization and Pose from Semantic Labellings](http://www2.maths.lth.se/vision/publdb/reports/pdf/toft-etal-iccv-2017.pdf)&nbsp;[2017 ICCV]

------

#### 6D Object Pose Estimation

##### PVNet
[PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation](https://arxiv.org/abs/1812.11788)&nbsp;[2018 arXiv]&nbsp;[code: [code]()]&nbsp;[[project page](https://zju-3dv.github.io/pvnet/)]

##### PoseCNN
[PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes](https://arxiv.org/abs/1711.00199)&nbsp;[2018 RSS]&nbsp;[code: [tensorflow](https://github.com/yuxng/PoseCNN)]&nbsp;[[project page](https://rse-lab.cs.washington.edu/projects/posecnn/)]

#### Camera Relocalization
posenet

|                              | input             | arch                       | output                 | scenes                                                    |
| ---------------------------- | ----------------- | -------------------------- | ---------------------- | --------------------------------------------------------- |
| VLocNet++                    | 2 RGB<br />Images | ResNet-50                  | p + q<br />vo<br />seg | 7 Scenes<br />deeploc                                     |
| VLocNet                      | 2 RGB<br />Images | ResNet-50                  | p + q<br />vo          | 7 Scenes<br />Cambridge Landmarks(no Street)              |
| MapNet                       | RGB<br />Videos   | ResNet-34                  | p + log(q)             | 7 Scenes<br />Oxford RobotCar                             |
| NNnet                        |                   |                            |                        |                                                           |
| Hourglass Pose               | RGB<br />Images   | ResNet-34<br />+ Hourglass | p + q                  | 7 Scenes                                                  |
| VidLoc                       | RGB <br />Videos  | GoogLeNet<br />+ Bi-LSTM   | p                      | 7 Scenes<br />Oxford RobotCar                             |
| LSTM PoseNet                 | RGB<br />Images   | GoogLeNet<br />+ LSTM      | p + q                  | 7 Scenes<br />Cambridge Landmarks(no Street)<br />TUM-LSI |
| Geometric loss <br />PoseNet | RGB<br />Images   | GoogLeNet                  | p + q                  | 7 Scenes<br />Cambridge Landmarks<br />Dubrovnik6K        |
| Bayesian<br />PoseNet        | RGB<br />Images   | GoogLeNet                  | p + q                  | 7 Scenes<br />Cambridge Landmarks                         |
| PoseNet                      | RGB<br />Images   | GoogLeNet                  | p + q                  | 7 Scenes<br />Cambridge Landmarks                         |

##### VLocNet++

[VLocNet++: Deep Multitask Learning for Semantic Visual Localization and
Odometry](https://arxiv.org/abs/1804.08366)&nbsp;[2018 RAL]

##### VLocNet
[Deep Auxiliary Learning for Visual Localization and Odometry](https://arxiv.org/abs/1803.03642)&nbsp;[2018 ICRA]

##### MapNet 
[ Geometry-Aware Learning of Maps for Camera Localization](https://arxiv.org/abs/1712.03342)&nbsp;[2018 CVPR]&nbsp;[code: [pytorch](https://github.com/NVlabs/geomapnet)]&nbsp;[[project page](https://research.nvidia.com/publication/2018-06_Geometry-Aware-Learning-of)]

##### LessMore
[Learning Less is More - 6D Camera Localization via 3D Surface Regression](https://arxiv.org/abs/1711.10228) [2018 CVPR]&nbsp;[code: [torch](https://github.com/vislearn/LessMore)]&nbsp;[[project page](https://hci.iwr.uni-heidelberg.de/vislearn/research/scene-understanding/pose-estimation/#CVPR18)]

##### NNnet 
[Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network](https://arxiv.org/abs/1707.09733) [2017 ICCV] &nbsp;[code: [torch](https://github.com/AaltoVision/camera-relocalisation)]

##### Geometric loss PoseNet 
[Geometric loss functions for camera pose regression with deep learning](https://arxiv.org/abs/1704.00390)&nbsp;[2017 CVPR]

##### Hourglass Pose
[Image-based Localization using Hourglass Networks](https://arxiv.org/abs/1703.07971)&nbsp;[2017 ICCV]

##### VidLoc
[VidLoc: A Deep Spatio-Temporal Model for 6-DoF Video-Clip Relocalization](https://arxiv.org/abs/1702.06521)&nbsp;[2017 CVPR]

##### LSTM PoseNet
[Image-based localization using LSTMs for structured feature correlation](https://arxiv.org/abs/1611.07890)&nbsp;[2017 ICCV]&nbsp;[code: [pytorch](https://github.com/hazirbas/poselstm-pytorch)]

##### DSAC
[DSAC - Differentiable RANSAC for Camera Localization](https://arxiv.org/abs/1611.05705)&nbsp;[2017 CVPR]&nbsp;[code: [torch](https://github.com/cvlab-dresden/DSAC)]&nbsp;[[project page](https://hci.iwr.uni-heidelberg.de/vislearn/research/scene-understanding/pose-estimation/#DSAC)]

##### BranchNet
[Delving Deeper into Convolutional Neural Networks for Camera Relocalization](http://www.xlhu.cn/papers/Wu17-icra.pdf)&nbsp;[2017 ICRA]

##### Bayesian PoseNet
[Modelling Uncertainty in Deep Learning for Camera Relocalization](https://arxiv.org/abs/1509.05909)&nbsp;[2016 ICRA]

##### PoseNet 
[PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://arxiv.org/abs/1505.07427)&nbsp;[2015 ICCV]&nbsp;[code: [caffe](https://github.com/alexgkendall/caffe-posenet)]&nbsp;[[project page](http://mi.eng.cam.ac.uk/projects/relocalisation/)]

##### *Benchmark* 

Data

[Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/)<br />
[7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)<br />
[Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/)<br />
[Dubrovnik6K and Rome16K ](http://www.cs.cornell.edu/projects/p2f/)<br />

------

#### VO

##### MagicVO
[MagicVO: End-to-End Monocular Visual Odometry through Deep Bi-directional
Recurrent Convolutional Neural Network](https://arxiv.org/abs/1811.10964)&nbsp;[2018 arXiv]

##### UnDeepVO 
[UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning](https://arxiv.org/abs/1709.06841)&nbsp;[2018 arXiv]&nbsp;[[project page](http://senwang.gitlab.io/UnDeepVO/)]

##### DeepVO 
[DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429)&nbsp;[2017 ICRA]&nbsp;[[project page](http://senwang.gitlab.io/DeepVO/)] 

------

#### VIO

##### VINet
[VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem](https://arxiv.org/abs/1701.08376)&nbsp;[2017 AAAI]

------

#### SFM

##### GeoNet
[GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276)&nbsp;[2018 CVPR]&nbsp;[code: [tensorflow](https://github.com/yzcjtr/GeoNet)]

##### SfMLearner
[Unsupervised Learning of Depth and Ego-Motion from Video](https://arxiv.org/abs/1704.07813)&nbsp;[2017 CVPR]&nbsp;[code: [tensorflow](https://github.com/tinghuiz/SfMLearner)]&nbsp;[[project page](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)]

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

##### Deep SLAM
[Toward Geometric Deep SLAM](https://arxiv.org/abs/1707.07410)&nbsp;[2017 arXiv]

------

#### Semantic SLAM

##### CNN-SLAM
[CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction](https://arxiv.org/abs/1704.03489)&nbsp;[2017 CVPR]

------

#### Depth Estimation

##### monoDepth
[Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677)&nbsp;[2017 CVPR]&nbsp;[code: [tensorflow](https://github.com/mrharicot/monodepth)]&nbsp;[[project page](http://visual.cs.ucl.ac.uk/pubs/monoDepth/)]

##### Unsupervised Depth Estimation
[Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue](https://arxiv.org/abs/1603.04992)&nbsp;[2016 ECCV]&nbsp;[code: [caffe](https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation)]

------

#### Flow Estimation

##### FlowNet2.0
[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925)&nbsp;[2017 CVPR]&nbsp;[code [caffe](https://github.com/lmb-freiburg/flownet2)]

##### FlowNet
[FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)&nbsp;[2015 ICCV]&nbsp;[code: [caffe](https://lmb.informatik.uni-freiburg.de/resources/binaries/dispflownet/dispflownet-release-1.2.tar.gz)]&nbsp;[[project page](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/)]

------

#### Visual Navigation

##### CMP
[Cognitive Mapping and Planning for Visual Navigation](https://arxiv.org/abs/1702.03920)&nbsp;[2017 CVPR]

##### Target-driven Visual Navigation
[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](https://arxiv.org/abs/1609.05143)&nbsp;[2017 ICRA]&nbsp;[code: [tensorflow](https://github.com/zfw1226/icra2017-visual-navigation)]

------

#### Local Feature

|               | detect | desc |
| ------------- | ------ | ---- |
| GeoDesc       |        |      |
| LF-NET        |        |      |
| SIPS          |        |      |
| DOAP          |        |      |
| SuperPoint    | Y      | Y    |
| AffNet        |        |      |
| HardNet       |        |      |
| Quad-networks |        |      |
| UCN           |        |      |
| LIFT          |        |      |
| DeepDesc      |        |      |
| DeepCompare   |        |      |
| TILDE         |        |      |
| MatchNet      |        |      |



##### GeoDesc
[GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://arxiv.org/abs/1807.06294)&nbsp;[2018 ECCV]&nbsp;[code: [tensorflow](https://github.com/lzx551402/geodesc)]

##### LF-NET
[LF-Net: Learning Local Features from Images](https://arxiv.org/abs/1805.09662)&nbsp;[2018 NIPS]&nbsp;[code: [tensorflow](https://github.com/vcg-uvic/lf-net-release)]

##### SIPS
[SIPS: Unsupervised Succinct Interest Points](https://arxiv.org/abs/1805.01358)&nbsp;[2018 arXiv]

##### DOAP
[Local Descriptors Optimized for Average Precision](https://arxiv.org/abs/1804.05312)&nbsp;[2018 CVPR][code: [matconvnet](http://cs-people.bu.edu/hekun/papers/DOAP/index.html)]&nbsp;[[project page](http://cs-people.bu.edu/hekun/papers/DOAP/index.html)]

##### SuperPoint
[SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)&nbsp;[2017 arXiv]&nbsp;[code: [pytorch](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)]

##### AffNet
[Repeatability Is Not Enough: Learning Affine Regions via Discriminability](https://arxiv.org/abs/1711.06704)&nbsp;[2018 ECCV][code: [pytorch](https://github.com/ducha-aiki/affnet)]

##### HardNet
[Working hard to know your neighbor’s margins: Local descriptor learning loss](https://arxiv.org/abs/1705.10872)&nbsp;[2017 NIPS][code: [pytorch](https://github.com/DagnyT/hardnet)]

##### Quad-networks
[Quad-networks: unsupervised learning to rank for interest point detection](https://arxiv.org/abs/1611.07571)&nbsp;[2017 CVPR]

##### UCN
[Universal Correspondence Network](https://arxiv.org/abs/1606.03558)&nbsp;[2016 NIPS]

##### LIFT
[LIFT: Learned Invariant Feature Transform](https://arxiv.org/abs/1603.09114)&nbsp;[2016 ECCV]

##### DeepDesc
[ Discriminative Learning of Deep Convolutional Feature Point Descriptors](https://icwww.epfl.ch/~trulls/pdf/iccv-2015-deepdesc.pdf)&nbsp;[2015 ICCV]

##### DeepCompare
[Learning to Compare Image Patches via Convolutional Neural Networks](https://arxiv.org/abs/1504.03641)&nbsp;[2015 CVPR]&nbsp;[code: [torch](https://github.com/szagoruyko/cvpr15deepcompare)]&nbsp;[[project page](http://imagine.enpc.fr/~zagoruys/publication/deepcompare/)]

##### TILDE
[ TILDE: A Temporally Invariant Learned DEtector](https://arxiv.org/abs/1411.4568)&nbsp;[2015 CVPR]

##### MatchNet
[MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Han_MatchNet_Unifying_Feature_2015_CVPR_paper.pdf)&nbsp;[2015 CVPR]&nbsp;[code: [caffe](https://github.com/hanxf/matchnet)]

##### *Evaluation* 
Protocol

[A performance evaluation of local descriptors](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_pami2004.pdf)&nbsp;[2005 PAMI]

Data

[HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors](https://arxiv.org/abs/1704.05939)&nbsp;[2017 CVPR]

------

#### 3D Net

##### PointSift
[PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation](https://arxiv.org/abs/1807.00652)&nbsp;[2018 arXiv]&nbsp;[code: [tensorflow](https://github.com/MVIG-SJTU/pointSIFT)]

##### PointCNN
[PointCNN: Convolution On X -Transformed Points](https://arxiv.org/abs/1801.07791)&nbsp;[2018 NIPS]&nbsp;[code: [tensorflow](https://github.com/yangyanli/PointCNN)]

##### PointNet++
[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)&nbsp;[2017 NIPS]&nbsp;[code: [tensorflow](https://github.com/charlesq34/pointnet2)]&nbsp;[[project page](http://stanford.edu/~rqi/pointnet2/)]

##### PointNet
[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)&nbsp;[2017 CVPR]&nbsp;[code: [tensorflow](https://github.com/charlesq34/pointnet)]&nbsp;[[project page](http://stanford.edu/~rqi/pointnet/)]


