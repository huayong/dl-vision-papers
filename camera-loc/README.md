#### Contents  
- [Retrieval Based](#retrieval-based)
- [Semantic Based](#semantic-based)
- [Local Feature Based](#local-feature-based)
- [PoseNet Based](#posenet-based)

------

------

| Scene      | [Active Search](#active-search) | [SCoRe Forest](#score-forest) | PoseNet | Bayesian PoseNet |      |      |      | VLocNet |      |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------- | ---------------- | ---- | ---- | ---- | ------- | ---- |
| Chess      | $0.04m, 1.96^ \circ$                                         | $0.03m, \mathbf{0.66^ \circ}$                                |         |                  |      |      |      |         |      |
| Fire       | $0.03m, 1.53^ \circ$                                         | $0.05m, 1.50^ \circ$                                         |         |                  |      |      |      |         |      |
| Heads      | $0.02m, 1.45^ \circ$                                         | $0.06m, 5.50^ \circ$                                         |         |                  |      |      |      |         |      |
| Office     | $0.09m, 3.61^ \circ$                                         | $0.04m, 0.78^ \circ$                                         |         |                  |      |      |      |         |      |
| Pumpkin    | $0.08m, 3.10^ \circ$                                         | $0.04m, \mathbf{0.68^ \circ}$                                |         |                  |      |      |      |         |      |
| RedKitchen | $0.07m, 3.37^ \circ$                                         | $0.04m, \mathbf{0.76^ \circ}$                                |         |                  |      |      |      |         |      |
| Stairs     | $0.03m, 2.22^ \circ$                                         | $0.04m, 1.32^ \circ$                                         |         |                  |      |      |      |         |      |
| *Average*  | $0.05m, 2.46^ \circ$                                         | $0.08m, 1.60^ \circ$                                         |         |                  |      |      |      |         |      |

​        median localization error compare on the microsoft 7-scenes dataset

#### Retrieval Based

##### RelocNet
[RelocNet: Continuous Metric Learning Relocalisation using Neural Nets](http://openaccess.thecvf.com/content_ECCV_2018/papers/Vassileios_Balntas_RelocNet_Continous_Metric_ECCV_2018_paper.pdf)&nbsp;[2018 ECCV]&nbsp;[[project page](http://relocnet.avlcode.org/)]

> 1. 和[NNnet](#nnnet)类似，也是先retrieval最相似图像在回归relative pose，最后得到最终的reloc pose；
> 2. 本文主要的贡献主要是求一个可以衡量camera movement的image descriptor，衡量的不仅是图像的相识度，重要的是衡量图像直接camera pose的相似度；NNnet里面直接使用一个relative pose的loss去push网络学习，而本文则是计算两帧图像camera的frustum overlap distance，push两帧图像descriptor之间的distance尽可能和该距离相同；

##### NNnet 
[Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network](https://arxiv.org/abs/1707.09733) [2017 ICCV] &nbsp;[code: [torch](https://github.com/AaltoVision/camera-relocalisation)]
> 1. 用CNN提取图像的描述特征，然后为了push学习到，利用两张图像的特征相减，继续利用CNN学习relative pose；
> 2. 利用检索的方法在map中查找和query image最相似的图像，求取relative pose；
> 3. 然后累加上相似图形的pose得到最终的pose；

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

> 主要思路：
> 
> 本文和[Semantic Labellings](#semantic-labellings)一个作者，也是利用语义信息去辅助long-term定位问题。这里是在传统2d-3d的基础上利用语义信息的一致性对每个匹配进行打分，分值当做权值进行传统的2d-3d求解。
> 
> 主要流程：
> 
> 1. 2d-3d匹配pose估计值生成阶段：在2d-3d描述符计算matching过程中，对每个2d-3d的匹配关系都生成了一系列的pose估计值；假设query图像局部相机坐标系的重力方向g是已知的，

##### Semantics-aware Visual Localization
[Semantics-aware Visual Localization under Challenging Perceptual Conditions](https://lmb.informatik.uni-freiburg.de/Publications/2017/OB17/naseer17icra.pdf)&nbsp;[2017 ICRA]

##### Semantic Labellings
[Long-term 3D Localization and Pose from Semantic Labellings](http://www2.maths.lth.se/vision/publdb/reports/pdf/toft-etal-iccv-2017.pdf)&nbsp;[2017 ICCV]

> 主要思路：
>
> ​        只利用图像语义分割label信息来进行定位，为了解决不同天气、季节以及光照等变化带来的long-term定位的问题，此时利用local feature的2d-3d方法明显是不可靠的了。
> 输入只有query image的语义label，地图中存储的是3D点、curves和点的语义信息；
>
> 主要流程：
>
> 1. 基于每张图像（地图和query） 的语义分割结果上半部分分成2x3的6个区域，每个区域进行label数目直方图统计以及计算building和vegetation二值语义图的梯度直方图，归一化拼接成一个向量当做image的描述符；用该描述符可以进行图像检索，检索出的最近image的pose可以当做优化的初始的pose；(防止随机初始化陷入局部最优解)
> 2. 优化pose求解，这个部分就是和之前不同的情况，之前是优化的projection error，现在优化的有两项，一个是地图3D点投影到image上和相同该点label的距离，但因为3d点是稀疏的，只用该项优化得不到精确的pose，另一个优化就是利用不同label之间的contour，建图会存储3d curves，保存的就是不同label之间的边界(不同类型的边界形式不一样)，文中提供了三种： poles(其实该label本身就是直线文中构建地图也是拟合的)，road的边界以及skyline等；其实最后就是一些线段，计算error方式一样，也是projection到图像上，和图像的语义label的边界距离；
>   在实际计算中，得到query图像的语义分割结果后，为每个label都构建了一个error map(出去动态物体，行人和车辆)，其实就是该位置距离最近的对应的label的距离(截断距离)，有了这些error map，3d点投影error就是对应label的error map投影位置上的值，3d curves进行积分，简单可以理解为sample一些点，也是取error，对于边界curve，error是对于的两个label的error map的和(这个是我的理解，文中没仔细讲，但是文中提到对于query image的操作只有建立error map)
>
> 试验结果：
> 1. 同源数据下，效果和2d-3d还是有点差距；(contour这种约束没有2d-3d那边精确)
> 2. 但在非同源数据下，也就是map和query数据是在不同条件下采集的，这样本文的效果还是不错的，没有像传统2d-3d退化的那么严重；(此时2d-3d可能很多情况下已经找不到正确的匹配关系了)
> 
> 该方法有三个问题：
> 1. 该方法比较依赖分割的精度，不仅仅是IOU还有contour上的精度，适用于各种条件场景下的语义分割本身也是一个比较难解的问题（相对于pose还是好一些）；
> 2. 该方法依赖不同语义间的contour进行精确定位，这个精度感觉有限，所以使用该方法的精度上限不高，但结合传统方法应该比较好；
> 3. 适用于语义比较丰富的场景，如果场景中语义信息不是特别丰富，假设可能只有两三类，这种情况下不太适用，主要因为单张图像上可能只有一种语义；

------

#### Local Feature Based

##### LessMore
[Learning Less is More - 6D Camera Localization via 3D Surface Regression](https://arxiv.org/abs/1711.10228) [2018 CVPR]&nbsp;[code: [torch](https://github.com/vislearn/LessMore)]&nbsp;[[project page](https://hci.iwr.uni-heidelberg.de/vislearn/research/scene-understanding/pose-estimation/#CVPR18)]

##### DSAC
[DSAC - Differentiable RANSAC for Camera Localization](https://arxiv.org/abs/1611.05705)&nbsp;[2017 CVPR]&nbsp;[code: [torch](https://github.com/cvlab-dresden/DSAC)]&nbsp;[[project page](https://hci.iwr.uni-heidelberg.de/vislearn/research/scene-understanding/pose-estimation/#DSAC)]

##### Active Search
[Efficient & effective prioritized matching for large-scale image-based localization](http://people.inf.ethz.ch/sattlert/publications/Sattler201XPAMI.pdf)&nbsp;[2017 TPAMI]

##### CSL
[City-Scale Localization for Cameras with Known Vertical Direction](http://120.52.51.15/www.maths.lth.se/vision/publdb/reports/pdf/svarm-enqvist-etal-pami-16.pdf)&nbsp;[2017 PAMI]

##### SCoRe Forest

[Scene coordinate regression forests for camera relocalization in rgb-d images](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Shotton_Scene_Coordinate_Regression_2013_CVPR_paper.pdf)&nbsp;[2013 CVPR]

##### Camera Pose Voting
[Camera Pose Voting for Large-Scale Image-Based Localization](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zeisl_Camera_Pose_Voting_ICCV_2015_paper.pdf)&nbsp;[2015 ICCV]&nbsp;[[note](https://github.com/huayong/dl-vision-papers/blob/master/camera-loc/notes/local-feature-based/camera-pose-voting.md)]

##### Structure-less Resection
[Structure from Motion Using Structure-less Resection](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Structure_From_Motion_ICCV_2015_paper.pdf)&nbsp;[2015 ICCV]

> 问题：
>
> 1. 首先基于2D-3D方法的前提是map中的3D点在2D图上是可见的，这样才能获得匹配关系进一步求解camera pose，但有种情况是由于重建不完全或者其他原因导致map中的3D点在2D图像中并不可见，但2D图像和地图中的2D图像是有overlap的；
>
>    这个很好理解，举个例子，比如三个物体O1,O2,O3在一张桌子上，我们拍了三张图I1,I2,I3，I1图像只包含O1和O2，I2图像只包含O2和O3，I3图像只包含O1和O3。这里利用其中任意两张图去构建map，然后利用另外一张去做2D-3D匹配求camera pose，比如利用I1,I2构建map，三角化得到3D点，因为两张图像只有O2部分由overlap，此时只能构建O2的点，但是I3是看不到O2的，所以此时没法使用2D-3D求解；
>
>    但其实I3和I1,I2都是存在图像的overlap，采用2D-2D方法是能得到camera pose的；
>
>    基于上面的思想当然问题可能不仅仅是像上面的例子一样，3D本身构建过程中就会丢掉一部分信息，而且最全的信息还是来自原始图像，所以本文采用2D-2D的策略；
>
> 思路：
>
> 利用2D-2D匹配关系增量式的构建SFM，和原来2D-3D不同的是该论文只利用了图像间的2D-2D的匹配关系来求解camera pose；
>
> 1. 而且这里构建的是多相机之间的2D-2D的匹配关系，目的是为了防止下面这张情况，比如query图像如map中的图像可以完全overlap，但是和每张图像都是部分overlap，可能map中5张（举例）可以覆盖住query图像，暴力方法就是和每一张图像都进行2D-2D的匹配，本文想简化这个过程。
>

------

#### PoseNet Based

##### VLocNet++

[VLocNet++: Deep Multitask Learning for Semantic Visual Localization and Odometry](https://arxiv.org/abs/1804.08366)&nbsp;[2018 RAL]&nbsp;[[project page](http://deeploc.cs.uni-freiburg.de/)]

##### VLocNet
[Deep Auxiliary Learning for Visual Localization and Odometry](https://arxiv.org/abs/1803.03642)&nbsp;[2018 ICRA]&nbsp;[[project page](http://deeploc.cs.uni-freiburg.de/)]

##### MapNet 
[Geometry-Aware Learning of Maps for Camera Localization](https://arxiv.org/abs/1712.03342)&nbsp;[2018 CVPR]&nbsp;[code: [pytorch](https://github.com/NVlabs/geomapnet)]&nbsp;[[project page](https://research.nvidia.com/publication/2018-06_Geometry-Aware-Learning-of)]

##### Geometric loss PoseNet 
[Geometric loss functions for camera pose regression with deep learning](https://arxiv.org/abs/1704.00390)&nbsp;[2017 CVPR]

##### Hourglass Pose
[Image-based Localization using Hourglass Networks](https://arxiv.org/abs/1703.07971)&nbsp;[2017 ICCV]

##### VidLoc
[VidLoc: A Deep Spatio-Temporal Model for 6-DoF Video-Clip Relocalization](https://arxiv.org/abs/1702.06521)&nbsp;[2017 CVPR]

##### LSTM PoseNet
[Image-based localization using LSTMs for structured feature correlation](https://arxiv.org/abs/1611.07890)&nbsp;[2017 ICCV]&nbsp;[code: [pytorch](https://github.com/hazirbas/poselstm-pytorch)]

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
