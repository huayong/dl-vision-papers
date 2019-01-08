#### Contents  
- [Retrieval Based](#retrieval-based)
- [Semantic Based](#semantic-based)
- [PoseNet Based](#posenet-based)

------

------

#### Retrieval Based

##### CoarseToFine
[From Coarse to Fine: Robust Hierarchical Localization at Large Scale](https://arxiv.org/abs/1812.03506)&nbsp;[2018 arXiv]

> 1. 这篇工作是[HierarchicalLoc](#hierarchicalloc)的延续，基本思路一致，主要不同就是把2d-3d中的SIFT等传统描述符换成了现在比较火的deep feature，具体是[SuperPoint](#https://github.com/huayong/dl-vision-papers/tree/master/deep-feature#superpoint)；
> 2. 当然SuperPoint也是训练了一个轻量级mobile版本的，只不过是把NetVLAD一起来train；

##### HierarchicalLoc

[Leveraging Deep Visual Descriptors for Hierarchical Efficient Localization](https://arxiv.org/abs/1809.01019)&nbsp;[2018 CoRL]&nbsp;[code: [tensorflow](https://github.com/ethz-asl/hierarchical_loc)]

> 1. 利用image retrieval(轻量级mobile版本的[NetVLAD](https://github.com/huayong/dl-vision-papers/tree/master/deep-feature#netvlad))方式获取query image最相似的kfs；
> 2. 对获取的top k的kfs做covisibility clustering，其实思路比较简单，就是能观察到相同map中的3d points聚集成一类；
> 3. 利用上面kfs观察到的loca 3d points(local map)来做2d-3d的matching，这样可以直接做暴力的查找，而且是在相似的局部空间来做，效果比全图2d-3d要好；相当于利用retrieval代替了kd-tree或者BoW等加速方法（虽然这些可以加速，但同时会把inliers过滤掉，损害最后的精度），但是对精度影响不大；
> 3. 利用蒸馏方法在原有NetVLAD的基础上训练轻量级的；

------

#### Semantic Based

##### SIVO
[Visual SLAM with Network Uncertainty Informed Feature Selection](https://arxiv.org/abs/1811.11946)&nbsp;[2018 arXiv]&nbsp;[code: [caffe](https://github.com/navganti/SIVO)]

##### DeLS-3D
[DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map](https://arxiv.org/abs/1805.04949)&nbsp;[2018 CVPR]&nbsp;[code: [only test](https://github.com/pengwangucla/DeLS-3D)]

##### SVL
[Semantic Visual Localization](https://arxiv.org/abs/1712.05773)&nbsp;[2018 CVPR]

##### Semantically Segmented Images
[Long-term Visual Localization using Semantically Segmented Images](https://arxiv.org/abs/1801.05269)&nbsp;[2018 ICRA]

##### Semantic Match Consistency
[Semantic Match Consistency for Long-Term Visual Localization](http://people.inf.ethz.ch/sattlert/publications/Toft2018ECCV.pdf)&nbsp;[2018 ECCV]

##### Semantics-aware Visual Localization
[Semantics-aware Visual Localization under Challenging Perceptual Conditions](https://lmb.informatik.uni-freiburg.de/Publications/2017/OB17/naseer17icra.pdf)&nbsp;[2017 ICRA]

##### Semantic Labellings
[Long-term 3D Localization and Pose from Semantic Labellings](http://www2.maths.lth.se/vision/publdb/reports/pdf/toft-etal-iccv-2017.pdf)&nbsp;[2017 ICCV]

------

#### PoseNet Based

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

[VLocNet++: Deep Multitask Learning for Semantic Visual Localization and Odometry](https://arxiv.org/abs/1804.08366)&nbsp;[2018 RAL]&nbsp;[[project page](http://deeploc.cs.uni-freiburg.de/)]

##### VLocNet
[Deep Auxiliary Learning for Visual Localization and Odometry](https://arxiv.org/abs/1803.03642)&nbsp;[2018 ICRA]&nbsp;[[project page](http://deeploc.cs.uni-freiburg.de/)]

##### RelocNet
[RelocNet: Continuous Metric Learning Relocalisation using Neural Nets](http://openaccess.thecvf.com/content_ECCV_2018/papers/Vassileios_Balntas_RelocNet_Continous_Metric_ECCV_2018_paper.pdf)&nbsp;[2018 ECCV]&nbsp;[[project page](http://relocnet.avlcode.org/)]

##### MapNet 
[Geometry-Aware Learning of Maps for Camera Localization](https://arxiv.org/abs/1712.03342)&nbsp;[2018 CVPR]&nbsp;[code: [pytorch](https://github.com/NVlabs/geomapnet)]&nbsp;[[project page](https://research.nvidia.com/publication/2018-06_Geometry-Aware-Learning-of)]

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

[RelocDB](http://relocnet.avlcode.org/) from [RelocNet](#relocnet)

[DeepLoc](http://deeploc.cs.uni-freiburg.de/) from [VLocNet](#vLocnet)

[Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/) from [PoseNet](posenet)

[7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)

[Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/)

[Dubrovnik6K and Rome16K ](http://www.cs.cornell.edu/projects/p2f/)