## Camera Pose Voting

### 主要思路

### 基本流程

#### pose估计当做对齐问题处理

1. 定义 $R_g$ 为旋转矩阵，当前局部相机坐标系到重力对齐坐标系下(3d map已经是重力对齐的了)，此操作也相当于当前局部坐标系和地图坐标系的一个轴对齐，举个例子：就是之前的相机的 $z$ 轴是沿着 viewing ray 方向， 通过旋转 $R_g$，$z$ 轴是沿着重力方向（这里像上为正）；后面求解相机姿态简化为求绕重力方向的旋转 $R_\varphi \in R^{2 \times 2}$ 和位置 $t \in R^{3}$ 。
>这里想一下，如果 $R_g$ 是确定的已知的（传感器或者其他方式可以确定），那么其实确定的是camera的viewing ray和重力方向的夹角 $\beta$，这样相机的旋转两个旋转自由度可以确定，一个是绕着viewing ray轴旋转 $360^\circ$，另一个是始终viewing ray和重力方向的夹角 $\beta$ ，然后绕 $z$ 轴旋转 $360^\circ$，这样旋转轨迹也是一个圆锥。

2. 如果一个3d点 $X$ 投影到image上，与图像上的匹配点 $x$ 的距离在 $r$ 个pixels范围内，那么该2d-3d匹配关系就是一个inliers。这个关系反推也就是图像上2d点 $x$ 反投影到空间中，落在一个3d的error圆锥内。所以3d error圆锥在重力对齐的相机坐标系的公式为：
$$
c(x, r) = \nu \cdot \boldsymbol{r}(x + u), \forall{u} \in R^2, ||u|| = r, \nu \in R_{\geq{0}} \\ 
其中 \boldsymbol{r}(x) = R_g K^{-1}(x\;1)^T ,即反投影并到旋转到对齐坐标系
$$

3. 继续假设距离水平屏幕plane的高度 $-h$ 是已知的，也就是全局坐标系 $z = 0$平面，所以相机位置的 $z$ 值也知道了，就是 $z=-h$。

4. 通过上面的一些假设，可以得到，其中已知3d点 $X$ 和2d匹配点 $x$，以及inliers误差范围 $r$， 去求 $R_\varphi$ 和 ​$t^{\prime}$，：
$$
\begin{bmatrix}
      R_\varphi & 0 \\
      0  & 1 \\ 
      \end{bmatrix} X + 
\begin{bmatrix}
      t^{\prime} \\
      -h \\ 
      \end{bmatrix} \in c(x, r), \; t = [t^{\prime}, -h]
$$
同时可以看出此时的操作限定在圆锥截面上（在对齐重力的相机坐标系下），截面公式为：
$$
c_{z}(x, r) = X_z - h, \;\; X_z 为3d点X的z轴坐标(global坐标系)
$$
> 注意，截面是一个椭圆(只有viewing ray沿着重力方向才是正圆)，想象相机斜朝上看，此时看到的3d点 $X_z$ 都是正值（global坐标系），假设为5，此时相机距离水平面高度 $h=-3$，所以求解平面为 $z=2$ 平面与圆锥相交的那个椭圆平面（对齐重力的相机坐标系下）。
> 其中 $R_g$ 可以通过传感器或者vanishing point等求出来，但相机高度 $h$ 也是未知的，该文通过插值3d map中相机的位置得到地平面，此时query image距离地平面只有 $\pm5cm$ 的偏移。

5. 这里定义了 $[h_min,h_max]$ 作为 $h$ 的取值区间，那么对于原来的圆锥截面来说，变成了一小段圆锥区域。如果把这些区域头投影到地平面上，得到2d error shape（所有圆锥截面投影的合集）。这样就把 $h$ 的的不确定又转换成相机位置的问题，后面优化在2d error shape空间优化就可以了，如下图：

  ![2d-error-shape](https://github.com/huayong/dl-vision-papers/tree/master/camera-loc/notes/local-feature-based/camera-pose-voting-2d-error-shape.png)


#### 线性时间的pose voting

1. 上面第一步定义了在重力对齐的相机局部坐标系下的2d error shape，对齐问题又转换成了在2d空间下的相机的旋转和平移问题，然后每种变化（旋转和平移）可以统计投影到该变换对应的voting shape的3d点个数，但有个问题是，2d shape是无限的，这样需要尝试无限的平移取值。

   本文把这个2d error shape转换到global全局坐标系下，这个2d error shape在global全局坐标系下叫做voting shape。

2. 下面要做的就是把这个在重力对齐的相机坐标系下的2d error shape（而且已经投影到地平面上）转换成global坐标系下camera pose的位置的voting shape。此时另 $M$ 为一个匹配 $m = (x, X)$ 的2d error shape，$\overline{M}$ 为该shape的中心点，可以把该点认为是在相机坐标系下的 $X$ 在地平面上的投影（本身之前的shape就是 $X$ 在相机坐标系下投影到地平面上的取值区间）。

所以camera pose的error shape可以表示为，简单来说就是把反投影error转化成相机位置的不确定性（对该匹配来说，相机在位置在此范围内都是inliers），0表示相机位置在相机坐标系下是原点：
$$
M_C(m) = \{0 - p + \overline{M} \; | \; p \in M\}
$$
然后从相机坐标系到global世界坐标系的平移量就可以推导出来（这里不考虑 $z$ 轴），如下：
$$
t^{\prime} = X^{\prime} - \overline{M}
$$
然后推出相机的中心点在global世界坐标系中的voting shape为：
$$
V(m) = M_C(m) + t^{\prime}
$$
这里没有考虑旋转的因素，旋转局部相机坐标系角度 $\phi$，然后把旋转因素加上：
$$
t^{\prime}_\phi = X^{\prime} - R_{\phi}\overline{M} \\
V(m, \phi) = R_{\phi}M_C(m) + t^{\prime} = R_{\phi}M_C(m) + X^{\prime} - R_{\phi}\overline{M} = \{X^{\prime} - R_{\phi}p \; | \; p \in M\}
$$
然后变换旋转 $\phi$ 角度，得到每个匹配 $m$ 在global全局坐标系下的voting shape如下图：

![voting-shape](https://github.com/huayong/dl-vision-papers/tree/master/camera-loc/notes/local-feature-based/camera-pose-voting-voting-shape.png)



其实简单理解这个就是在某个camera pose下，2d点对应的inliers的3d的取值范围，此时匹配的3d点在global全局坐标系下的坐标是已知的，

### 问题

1. 如果此时相机的viewing ray是水平的，那么就没有圆锥截面了，此时应该怎么处理？？？？？
2. 而且随着相机的viewing ray越接近水平，搜索空间是增加，主要是因为水平面和3d error圆锥的中轴线越来越小，这样误差会增大？？？？？