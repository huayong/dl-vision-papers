#### Contents  
- [Global Feature](#global-feature)  
  - [DLEF](#dlef)
  - [PlaNet](#planet)
  - [NetVLAD](#netvlad)
  - [DenseVLAD](#densevlad)
- [Local Feature](#local-feature)  
  - [SuperPoint](#superpoint)
  - [GeoDesc](#geodesc)
  - [Quad-networks](#quad-networks)
  - [UCN](#ucn)
  - [LIFT](#lift)
  - [DeepDesc](#deepdesc)
  - [TILDE](#tilde)
  - [*Evaluation*](#evaluation)

------

------

#### Global Feature
or called Image Retrieval and Place Recognition and Image Representation and Image Desciptor

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
