#### 2D Vision

[Base Architecture](https://github.com/huayong/dl-vision-papers/tree/master/base-archs)

1. 基础的提取特征的分类框架，包括AlexNet、Googlenet、ResNet和DenseNet系列等；

[Segmentation Architecture](https://github.com/huayong/dl-vision-papers/tree/master/seg-archs)

1. 语义分割网络，包括Deeplab系列、PSPNet、SegNet和ENet等；
2. 实例分割网络；
3. 全景分割网络，融合了语义分割和实例分割；

[Object Detection Architecture](https://github.com/huayong/dl-vision-papers/tree/master/det-archs)

1. 检测网络，包括RCNN系列、YOLO系列、SSD系列等；

[Human Keypoint Detection Architecture](https://github.com/huayong/dl-vision-papers/tree/master/kps-archs)

1. 人体关节点检测网络，包括OpenPose、DensePose等；

[Multi-task Architecture](https://github.com/huayong/dl-vision-papers/tree/master/multi-archs)

1. 针对多个任务同时处理的网络架构；

[Mobile Architecture](https://github.com/huayong/dl-vision-papers/tree/master/mobile-archs)

1. 移动端模型框架，包括MobileNet和ShuffleNet系列等；

[NAS](https://github.com/huayong/dl-vision-papers/tree/master/nas-archs)
1. 模型框架自动学习，包括NASNet系列等；

#### 3D Vision

[3D Recon](https://github.com/huayong/dl-vision-papers/tree/master/3d-recon)

一般来说利用学习的方法进行重新，重建后的三维结构也包括三维语义的信息。

1. 场景重建；
2. 物体重建；
3. 平面重建；

[3D Vision](https://github.com/huayong/dl-vision-papers/tree/master/3d-vision)

1. SFM，利用网络恢复pose和depth等；
2. MVS，利用CNN网络恢复多帧depth等；
2. 学习方法应用到SLAM上，包括一些语义信息的辅助SLAM和动态场景下SLAM等；
2. VO，利用CNN求前后帧的Relative Pose；
3. VIO，结合IMU信息求前后帧的Relative Pose；
4. 利用CNN网络直接估计单帧图像Depth；
5. 利用CNN网络估计前后帧之间Flow信息；

[3D Data Architecture](https://github.com/huayong/dl-vision-papers/tree/master/3d-archs)

1. 点云为输入的模型框架，包括PointNet系列等；
2. Depth为输入的模型框架；
3. RGBD为输入的模型框架；

[6D Object Pose](https://github.com/huayong/dl-vision-papers/tree/master/6d-object-pose)

利用CNN网络估计单帧图像中物体 6-DoF 位姿。

1. 室内场景物体；
2. 室外场景物体，无人驾驶应用场景，大部分是车辆的位姿；

[Camera Loc](https://github.com/huayong/dl-vision-papers/tree/master/camera-loc)

1. 两段式先图像检索再进行2d-3d优化；
2. 直接网络回归相机姿态，包括posenet系列等；
3. 语义约束辅助的姿态估计；
4. 传统的基于2d-3d或者2d-2d匹配关系求解camera location；

[Deep Feature](https://github.com/huayong/dl-vision-papers/tree/master/deep-feature)

1. 图像全局描述符，一般用于图像检索（Image Retrieval）、地点识别( Place Recognition）等；
2. 图像局部描述符，包括特征点提取，描述符计算，匹配度量算法学习等；
3. 图像 appearance transfer，利用 gan 把特殊情况下（晚上，下雪）的图像转成一般情况处理，主要解决特殊情况下的图像匹配问题；

