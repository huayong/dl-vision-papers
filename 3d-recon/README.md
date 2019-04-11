#### Contents  
- [Scene Recon](#scene-recon)   
- [Object Recon](#object-recon)   
- [Plane Recon](#scene-recon)   

------

------

#### Scene Reon

##### Factored3d
[Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene](https://arxiv.org/abs/1712.01812)&nbsp;[2018 CVPR]&nbsp;[code: [pytorch](https://github.com/shubhtuls/factored3d)]&nbsp;[[project page](https://shubhtuls.github.io/factored3d/)]

##### Holistic Parsing
[Holistic 3D Scene Parsing and Reconstruction from a Single RGB Image](https://arxiv.org/abs/1808.02201)&nbsp;[2018 ECCV]&nbsp;[code: [python](https://github.com/thusiyuan/holistic_scene_parsing)]&nbsp;[[project page](http://siyuanhuang.com/holistic_parsing/main.html)]

##### SSCNet
[Semantic Scene Completion from a Single Depth Image](https://arxiv.org/abs/1611.08974)&nbsp;[2017 CVPR]&nbsp;[code: [caffe](https://github.com/shurans/sscnet)]&nbsp;[[project page](http://sscnet.cs.princeton.edu/)]

------

#### Object Recon

##### GenRe
[Learning to Reconstruct Shapes from Unseen Classes](https://arxiv.org/abs/1812.11166)&nbsp;[2018 NIPS(oral)]&nbsp;[code: [pytorch](https://github.com/xiumingzhang/GenRe-ShapeHD)]&nbsp;[[project page](http://genre.csail.mit.edu/)]

##### ShapeHD
[Learning Shape Priors for Single-View 3D Completion and Reconstruction](https://arxiv.org/abs/1809.05068)&nbsp;[2018 ECCV]&nbsp;[code: [pytorch](https://github.com/xiumingzhang/GenRe-ShapeHD)]&nbsp;[[project page](http://shapehd.csail.mit.edu/)]

##### MarrNet
[MarrNet: 3D Shape Reconstruction via 2.5D Sketches](https://arxiv.org/abs/1711.03129)&nbsp;[2017 NIPS]&nbsp;[code: [torch7](https://github.com/jiajunwu/marrnet)&nbsp;[pytorch](https://github.com/xiumingzhang/GenRe-ShapeHD)]&nbsp;[[project page](http://marrnet.csail.mit.edu/)]

##### 3D-GAN
[Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/abs/1610.07584)&nbsp;[2016 NIPS]&nbsp;[code: [torch7](https://github.com/zck119/3dgan-release)]&nbsp;[[project page](http://3dgan.csail.mit.edu/)]

------

#### Plane Recon

##### PlanarReconstruction
[Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding](https://arxiv.org/abs/1902.09777)&nbsp;[2019 CVPR]&nbsp;[code: [pytorch](https://github.com/svip-lab/PlanarReconstruction)]

> 1. 针对室内家居场景，ScanNet数据集；
> 2. 

##### PlaneRCNN
[PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image](https://arxiv.org/abs/1812.04072)&nbsp;[2019 CVPR Oral]

> 1. 室内室外双场景都适用；
> 2. 

##### PlaneRecover
[Recovering 3D Planes from a Single Image via Convolutional Neural Networks](https://arxiv.org/pdf/1812.04072)&nbsp;[2018 ECCV]&nbsp;[code: [tensorflow](https://github.com/fuy34/planerecover)]

> 1. 针对室外无人车城市道路场景；
> 2. 利用CNN网络预测plane参数，同时预测plane区域(不是语义分割)；

##### PlaneNet
[PlaneNet: Piece-wise Planar Reconstruction from a Single RGB Image](https://arxiv.org/abs/1804.06278)&nbsp;[2018 CVPR]&nbsp;[code: [tensorflow](https://github.com/art-programmer/PlaneNet)]&nbsp;[[project page](http://art-programmer.github.io/planenet.html)]

> 1. 针对室内家居场景；
> 2. 利用CNN网络预测plane参数，同时预测plane的depth和语义分割区域；