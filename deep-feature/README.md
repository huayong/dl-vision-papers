#### Contents  
- [Appearance Transfer](#appearance-transfer)
- [Global Feature](#global-feature)  
- [Local Feature](#local-feature) 

------

------

#### Appearance Transfer

##### ToDayGAN
[Night-to-Day Image Translation for Retrieval-based Localization](https://arxiv.org/abs/1809.09767)&nbsp;[2018 arXiv]&nbsp;[code: [pytorch](https://github.com/AAnoosheh/ToDayGAN)]

##### ATAC
[Adversarial Training for Adverse Conditions: Robust Metric Localisation using Appearance Transfer](https://arxiv.org/abs/1803.03341)&nbsp;[2018 CoRR]

##### PRGAN
[Addressing Challenging Place Recognition Tasks using Generative Adversarial Networks](https://arxiv.org/abs/1709.08810)&nbsp;[2018 ICRA]

------

#### Global Feature

##### DLEF
[ Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/pdf/1612.06321.pdf)&nbsp;[2017 ICCV]&nbsp;[code: [tensorflow](https://github.com/tensorflow/models/tree/master/research/delf)]

##### PlaNet
[PlaNet - Photo Geolocation with Convolutional Neural Networks](https://arxiv.org/abs/1602.05314)&nbsp;[2016 ECCV]

##### NetVLAD
[NetVLAD: CNN architecture for weakly supervised place recognition](https://arxiv.org/abs/1511.07247)&nbsp;[2016 CVPR]&nbsp;[code: [matconvnet](https://github.com/Relja/netvlad)]&nbsp;[[project page](https://www.di.ens.fr/willow/research/netvlad/)]

##### DenseVLAD
[24/7 place recognition by view synthesis](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/download/Torii-CVPR-2015-final.pdf)&nbsp;[2015 CVPR]&nbsp;[code: [c++](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/)]&nbsp;[[project page](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/)]

##### *Evaluation*

------

#### Local Feature

|                                  | detect | desc | type | length |
| -------------------------------- | ------ | ---- | ---- | ------ |
| [GeoDesc](#geodesc)              |        |      |      |        |
| [LF-NET](#lf-net)                 |        |      |      |        |
| [SIPS](#sips)                     |        |      |      |        |
| [DOAP](#doap)                     |        |      |      |        |
| [SuperPoint](#superpoint)         | Y      | Y    | real | 256    |
| [AffNet](#affnet)                 |        |      |      |        |
| [HardNet](#hardnet)               |        |      |      |        |
| [Spread-out](#spread-out)         |        |      |      |        |
| [DeepCD](#deepcd)                 |        |      |      |        |
| [Quad-networks](#quad-networks)   |        |      |      |        |
| [L2-Net](#l2-net)                 |        |      |      |        |
| [UCN](#ucn)                       |        |      |      |        |
| [LIFT](#lift)                     |        |      |      |        |
| [DeepPatchMatch](#deeppatchmatch) |        |      |      |        |
| [DeepBit](#deepbit)               |        |      |      |        |
| [TFeat](#tfeat)                   |        |      |      |        |
| [PN-Net](#pn-net)                 |        |      |      |        |
| [DeepDesc](#deepdesc)             |        |      |      |        |
| [DeepCompare](#deepcompare)       |        |      |      |        |
| [TILDE](#tilde)                   |        |      |      |        |
| [MatchNet](#matchnet)             |        |      |      |        |

##### GeoDesc
[GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://arxiv.org/abs/1807.06294)&nbsp;[2018 ECCV]&nbsp;[code: [tensorflow](https://github.com/lzx551402/geodesc)]&nbsp;[[note](https://blog.csdn.net/honyniu/article/details/86617082)]

##### LF-NET
[LF-Net: Learning Local Features from Images](https://arxiv.org/abs/1805.09662)&nbsp;[2018 NIPS]&nbsp;[code: [tensorflow](https://github.com/vcg-uvic/lf-net-release)]

##### SIPS
[SIPS: Unsupervised Succinct Interest Points](https://arxiv.org/abs/1805.01358)&nbsp;[2018 arXiv]

##### DOAP
[Local Descriptors Optimized for Average Precision](https://arxiv.org/abs/1804.05312)&nbsp;[2018 CVPR][code: [matconvnet](http://cs-people.bu.edu/hekun/papers/DOAP/index.html)]&nbsp;[[project page](http://cs-people.bu.edu/hekun/papers/DOAP/index.html)]

##### SuperPoint
[SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)&nbsp;[2017 arXiv]&nbsp;[code: [pytorch](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)]

##### AffNet
[Repeatability Is Not Enough: Learning Affine Regions via Discriminability](https://arxiv.org/abs/1711.06704)&nbsp;[2018 ECCV]&nbsp;[code: [pytorch](https://github.com/ducha-aiki/affnet)]

##### HardNet
[Working hard to know your neighbor’s margins: Local descriptor learning loss](https://arxiv.org/abs/1705.10872)&nbsp;[2017 NIPS]&nbsp;[code: [pytorch](https://github.com/DagnyT/hardnet)]

##### Spread-out
[Learning Spread-out Local Feature Descriptors](https://arxiv.org/abs/1708.06320)&nbsp;[2017 ICCV]&nbsp;[code: [tensorflow](https://github.com/ColumbiaDVMM/Spread-out_Local_Feature_Descriptor)]

##### DeepCD
[DeepCD: Learning Deep Complementary Descriptors for Patch Representations](https://www.csie.ntu.edu.tw/~cyy/publications/papers/Yang2017DLD.pdf)&nbsp;[2017 ICCV]&nbsp;[code: [tensorflow](https://github.com/shamangary/DeepCD)]

##### Quad-networks
[Quad-networks: unsupervised learning to rank for interest point detection](https://arxiv.org/abs/1611.07571)&nbsp;[2017 CVPR]

##### L2-Net
[L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space](http://www.nlpr.ia.ac.cn/fanbin/pub/L2-Net_CVPR17.pdf)&nbsp;[2017 CVPR]&nbsp;[code: [matconvnet](https://github.com/yuruntian/L2-Net)]

##### UCN
[Universal Correspondence Network](https://arxiv.org/abs/1606.03558)&nbsp;[2016 NIPS]

##### LIFT
[LIFT: Learned Invariant Feature Transform](https://arxiv.org/abs/1603.09114)&nbsp;[2016 ECCV]&nbsp;[code: [tensorflow](https://github.com/cvlab-epfl/tf-lift)&nbsp;[theano](https://github.com/cvlab-epfl/LIFT)]

##### TFeat
[Learning local feature descriptors with triplets and shallow convolutional neural networks](http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf)&nbsp;[2016 BMVC]&nbsp;[code: [pytorch](https://github.com/vbalnt/tfeat)]

##### PN-Net
[PN-Net: Conjoined Triple Deep Network for Learning Local Image Descriptors](https://arxiv.org/abs/1601.05030)&nbsp;[2016 arXiv]&nbsp;[code: [pytorch](https://github.com/vbalnt/pnnet)]

##### DeepPatchMatch
[Learning Local Image Descriptors with Deep Siamese and Triplet Convolutional Networks by Minimizing Global Loss Functions](https://arxiv.org/abs/1512.09272)&nbsp;[2016 CVPR]&nbsp;[code: [matconvnet](https://github.com/vijaykbg/deep-patchmatch)]

##### DeepBit
[Learning Compact Binary Descriptors with Unsupervised Deep Neural Networks](http://www.iis.sinica.edu.tw/~kevinlin311.tw/cvpr16-deepbit.pdf)&nbsp;[2016 CVPR]&nbsp;[code: [caffe](https://github.com/kevinlin311tw/cvpr16-deepbit)]

##### DeepDesc
[ Discriminative Learning of Deep Convolutional Feature Point Descriptors](https://icwww.epfl.ch/~trulls/pdf/iccv-2015-deepdesc.pdf)&nbsp;[2015 ICCV]

##### DeepCompare
[Learning to Compare Image Patches via Convolutional Neural Networks](https://arxiv.org/abs/1504.03641)&nbsp;[2015 CVPR]&nbsp;[code: [torch](https://github.com/szagoruyko/cvpr15deepcompare)]&nbsp;[[project page](http://imagine.enpc.fr/~zagoruys/publication/deepcompare/)]

##### TILDE
[ TILDE: A Temporally Invariant Learned DEtector](https://arxiv.org/abs/1411.4568)&nbsp;[2015 CVPR]

##### MatchNet
[MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Han_MatchNet_Unifying_Feature_2015_CVPR_paper.pdf)&nbsp;[2015 CVPR]&nbsp;[code: [caffe](https://github.com/hanxf/matchnet)]

##### *Evaluation* 

###### PhotoSync
[A Large Dataset for improving Patch Matching](https://arxiv.org/abs/1801.01466)&nbsp;[2018 arXiv]&nbsp;[[code](https://github.com/rmitra/PS-Dataset)]

###### ETH local features

[Comparative Evaluation of Hand-Crafted and Learned Local Features](https://www.cvg.ethz.ch/research/local-feature-evaluation/schoenberger2017comparative.pdf)&nbsp;[2017 CVPR]&nbsp;[[code](https://github.com/ahojnnes/local-feature-evaluation)]&nbsp;[[project page](https://cvg.ethz.ch/research/local-feature-evaluation/)]

###### HPatches
[HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors](https://arxiv.org/abs/1704.05939)&nbsp;[2017 CVPR]&nbsp;[[project page](https://hpatches.github.io/)]

###### Heinly
[Comparative Evaluation of Binary Features](http://rogerioferis.com/VisualRecognitionAndSearch2013/material/Class2EfficientFeatureSurvey.pdf)&nbsp;[2012 ECCV]&nbsp;[[project page](http://cs.unc.edu/~jheinly/binary_descriptors.html)]

###### Brown
[Discriminative Learning of Local Image Descriptors](http://matthewalunbrown.com/papers/pami2010.pdf)&nbsp;[2010 PAMI]&nbsp;[[project page](http://matthewalunbrown.com/patchdata/patchdata.html)]


[A performance evaluation of local descriptors](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_pami2004.pdf)&nbsp;[2005 PAMI]

