#### Contents  
- [Retrieval Based](#retrieval-based)
- [Semantic Based](#semantic-based)
- [PoseNet Based](#posenet-based)

------

------

#### Retrieval Based

##### CoarseToFine
[From Coarse to Fine: Robust Hierarchical Localization at Large Scale]()&nbsp;[2018 arXiv]

##### HierarchicalLoc
[Leveraging Deep Visual Descriptors for Hierarchical Efficient Localization]()&nbsp;[2018 CoRL]

------

#### Semantic Based

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

[Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/)<br />
[7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)<br />
[Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/)<br />
[Dubrovnik6K and Rome16K ](http://www.cs.cornell.edu/projects/p2f/)<br />
