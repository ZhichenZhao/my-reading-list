# Waterbro's Reading List on Computer Vision
这份笔记分为每种topic以及topic下的论文，topic按照字典序排列，论文按照时间顺序排列。

## Action Detection
### A Pursuit of Temporal Accuracy in General Activity Detection, arXiv,2017
* 先用TSN判断actionness，然后用规则得到group，最后同时判断动作和完整性

### Temporal Action Detection with Structured Segment Networks，CVPR，2017
* 以proposal为输入，同时考虑这个动作的开始阶段和结束阶段，并且做时间金字塔，最后合并成一个描述

### Weakly Supervised Action Localization by Sparse Temporal Pooling Network，arxiv，2017
* 两件事情挺有意思，一是他接一层两fc再sigmoid到0-1之间然后乘回去，这样当作一个attention map
二是下面通过推导把时间的那个系数给换出来了

### Action Tubelet Detector for Spatio-Temporal Action Localization，ICCV，2017
* stack连续几帧的feature，一起来过分类器出分出框，在frame level和video level上都是目前最好

### Rethinking the Faster-RCNN Architecture for Temporal Action Localization，CVPR，2018
* 思想是把fasterRCNN应用到动作的localization上，感觉很合理。
* 实际操作中，图片里检测物体，物体的scale不会变得太大，但是动作的时长变化却很大，为了解决这个问题，用K个子网络，每个子网络不一样，而且就是为了获得指定感受野大小的序列特征。
* 另一个方面是capture context，本来conv就是dilation的，那么context就是简单的二倍dilation就好了。
---

## Action Recognition
### Temporal Segment Networks: Towards Good  Practices for Deep Action Recognition， ECCV， 2016
* 从视频中平均的抽一些帧来联合地训练，是提升性能最重要的地方。
* 还用到了warped optical flow，移除掉了镜头移动的影响
* 可视化工具：Deep Draw

### Spatiotemporal Residual Networks for Video Action Recognition， NIPS，2016
* 为了解决spatial和motion分离的问题，在两支网络中间加入了一个connection。

### ConvNet Architecture Search for Spatiotemporal Feature Learning，arXiv，2017
* 把C3D改成了resnet的样子，在只用rgb的结果里面是最好的。

### ActionVLAD: Learning spatio-temporal aggregation for action classification,CVPR,2017
* 在卷积特征提取之后用VLAD进行空间和时间的双重解码，比pooling的方法好很多，但是把词典放在外面之后整个也难以endtoend，可以留做备选参考。
* 另一点是用到了late fusion，即rgb和光流分别做自己的词典表达，然后再接softmax。

### Attentional Pooling for Action Recognition，NIPS，2017
* 通过数学建模后面几层然后推导来进行attention map，虽然看起来操作很多，但是实际上和最普通的attetionmap实现无异，且实验有点投机取巧

### Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset，CVPR，2017
* 新的数据集，在这上面刷的很高

### Action Recognition with Coarse-to-Fine Deep Feature Integration and Asynchronous Fusion，AAAI，2018
* 主要用了多尺度的网络，有个adaptive挺有意思，conv3用top3的action监督，conv4用top2的，conv5用gt监督，这个东西看起来挺有启发意义的。后面的从粗到细的特征用LSTM来输出。在UCF上到了95.2

### ECO: Efficient Convolutional Network for Online Video Understanding，ECCV，2018
* idea挺好的，3D卷积效果好（只用rgb的情况下），但是太慢，但是3D的好往往在深层才能体现出来，于是先用2D的提一堆帧的特征，然后叠起来再过3D网络，是两种网络的一个权衡，性能保持的还不错的情况下速度提升很快。

---
## Attrubutes
### Deep Learning Face Attributes in the Wild，ICCV，2015
* 只是用CNN刷了一下attribute的库，没有其他亮点
### Age and Gender Classification using Convolutional Neural Networks，CVPR，2015
* 提到了属性网络用训好的人脸识别网络来初始化，以及要local conv

---
## Auto Driving
### VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition， ICCV，2017
* 有一些实践经验，比如在训练灭点的时候，如果只画一个圈圈，那么由于前背景像素数量差距较大，训练端的loss很快就会不下降，这里做了一个变形是把灭点做成四个四分之一区域的交集，这样就好了。

---
## Base model
### Rectifier Nonlinearities Improve Neural Network Acoustic Models，ICML，2013
* 提出了leaky RELU，为了防止一旦进入小与0就回不来，永远不激活的情况，可以让负的那一段又一个很小的斜率。

### Network in Network，ICLR，2014
* 结构是用多层的全连接的堆叠来替代卷积元，每一层都有激活，在最后面，不是通过全连接来输出，而是通过全局的池化来输出。
* 实际上第一步是变相的在卷积层中抬高了深度和非线性
* 后面一步的详细过程是生成很多头通道的feature map，然后pooling成一个向量再接softmax输出。
* 其实这里的mlpconv中，第一层是正常的，后面2,3层都是1x1的，这样通道数可以自由变换。
* 设想：如果一个网络全部是1x1的卷积和某种spatial context结合的关系得到，那么某型的数量将会小非常多，这是做小模型的关键。
* 而神经网络想要做快inference speed 主要依靠的是matrix decomposition。

### Spatial Transformer Network，NIPS，2015
* 它主要是说，尽管 CNN 一直号称可以做 spatial invariant feature extraction，但是这种 invariant 是很有局限性的。因为 CNN 的 max-pooling 首先只是在一个非常小的、rigid 的范围内（2×2 pixels）进行，其次即使是 stacked 以后，也需要非常 deep 才可以得到大一点范围的 invariant feature，三者来说，相比 attention 那种只能抽取 relevant 的 feature，我们需要的是更广范围的、更 canonical 的 features。为此它们提出了一种新的完全 self-contained transformation module，可以加入在网络中的任何地方，灵活高效地提取 invariant image features.
* 具体上，这个 module 就叫做 Spatial Transformers，由三个部分组成： Localization Network, Grid generator 和 Sampler。Localization Network 非常灵活，可以认为是一个非常 general 的进一步生成 feature map 和 map 对应的 parameter 的网络。因此，它不局限于用某一种特定的 network，但是它要求在 network 最后有一层 regression，因为需要将 feature map 的 parameter 输出到下一个部分：Grid generator。Grid generator 可以说是 Spatial Transformers 的核心，它主要就是生成一种“蒙版”，用于“抠图”（Photoshop 附体……）。Grid generator 定义了 Transformer function，这个 function 的决定了能不能提取好 invariant features。如果是 regular grid，就好像一张四四方方没有倾斜的蒙版，是 affined grid，就可以把蒙版“扭曲”变换，从而提取出和这个蒙版“变换”一致的特征。在这个工作中，只需要六个参数就可以把 cropping, translation, rotation, scale and skew 这几种 transformation 都涵盖进去，还是很强大的；而最后的 Sampler 就很好理解了，就是用于把“图”抠出来。

### Batch normalization: Accelerating deep net- work training by reducing internal covariate shift. arXiv, 2015
* 动机是当前一层在训练时，对于下一层来说数据的分布总是在改变，这样的话下一层就不太好训练。现在batch normalization是在同一个batch内做归一化。
* 统计机器学习有一个基本假设，源空间和目标空间的数据分布是一致的。如果不一致就需要transfer learning等手段。
* http://blog.csdn.net/happynear/article/details/44238541
* 为什么不在RELU之后归一化？按照第一章的理论，应当在每一层的激活函数之后，例如ReLU=max(Wx+b,0)之后，对数据进行归一化。然而，文章中说这样做在训练初期，分界面还在剧烈变化时，计算出的参数不稳定，所以退而求其次，在Wx+b之后进行归一化。因为初始的W是从标准高斯分布中采样得到的，而W中元素的数量远大于x，Wx+b每维的均值本身就接近0、方差接近1，所以在Wx+b后使用Batch Normalization能得到更稳定的结果。

### Learning Deep Embeddings with Histogram Loss，NIPS，2016
* 设计了histogram loss来对正样本和负样本的距离进行监督，类似于在一个batch内对整个batch进行tripletloss，按照他的说法，不需要两两比对，效率更高，但是对比直方图那里没看懂，为什么不直接对两个分布求相关呢？

### Sqeeze and Excitation Network, CVPR, 2017
* ImageNet2017的冠军，核心思想就是对每一个channel，不管spatial如何来用一个系数衡量它的重要性。

### Rethinking the Inception Architecture for Computer Vision,CVPR,2016
* inception v3模型，设计思想：
* 在最开始网络网络还比较浅的时候特征纬度不能太低，不然信息和内容全都被丢弃了。
* 把一个5x5的卷积用两个3x3的卷积代替,所以比3更大的卷积元就没有必要了。

### Local Binary Convolutional Neural Networks，CVPR，2017
* 从LBP中得到启发，直接把filter变成差分项，然后搭建一套可训练的系统网络，其实也可以理解为sift融进了网络里面，这样的设计可以极大的减少网络参数。

### Xception: Deep Learning with Depthwise Separable Convolutions，arXiv，2017
* 在inception的基础上，彻底分离卷积中空间的相关性和channel的相关性，每一层输入进来，先做一个1x1的conv再做localconv，他的conv是depthwise即channelwise的conv，按照single crop来测效果上优于resnet152和Inception。

### ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices，arXiv，2017
* 思路很有意思，如果不做group conv模型很大，如果极限情况每个channel自己做，由于看不到其他channel，性能损失很多。于是想到channel shuffle一下，然后还是按着对应关系卷，这样就变相的做了折中。

### SWISH: A SELF-GATED ACTIVATION FUNCTION,arXiv,2017
* 作者认为，这个swish激活函数有几个优点，无上下界，不单调，且足够光滑。没有上下界是比较重要的，这样模型不需要很小心地初始化才能达到一个比较好的初值。但是好像没有比softplus有优势？

### Dynamic Routing Between Capsules，NIPS，2017
* hinton对于CNN的替换idea，CNN中输出是响应值，然后融合方式是maxpooling，对于capsule来说，输出是向量，融合方式是路由选择，routing-by-agreement。
* capsule最本质的思想是要把输出变成向量，这样一方面是向量的方向很自由，不用浪费大量的空间去留给响应应该为0的类别。另一方面，在向量为目标下，各个类别可以建立一定的拓扑关系，最终结果是那个类别是看在那个方向上长度长不长。

### Shake-Shake regularization, submitted to ICLR 2018
* 对于两个分支的这种residual结构，在forward的时候用不同的系数乘以下，加起来是1,backward的时候也用不同的系数乘来传梯度，可以当作另一种dropout的方法，在cifar10,100上效果很好，但有一个问题是它所需要的epoch太长了（1800）

### FRACTALNET: ULTRA-DEEP NEURAL NETWORKS WITHOUT RESIDUALS，ICLR，2017
* 是对resnet的一种阐述和改进，只不过占用空间是个问题。在cifar上是最好的结果。

### Residual Connections Encourage Iterative Inference， ICLR，2018
* 对于resnet，有几个观察和探索。说在resnet中，浅层做representation learning，而深层则是feature refinement。为了证明第二点，有两个方面，第一是看h的相对幅度变化，刚开始很大，到后面慢慢变小，第二个实验是去掉最后的一层看看准确率会不会掉很多。在这一点的基础上想到后面的layer是可以share的，但是bn需要用recurrent的。这样下来resnet101与38参数量相当，还能涨一点点。

### Label Refinery: Improving ImageNet Classification through Label Progression，arXiv，2018
* 总结了分类方面目前留下的主要问题：1.同一张图片里面可能存在多个物体，直接认为就属于哪一个类实际上是反自然的，比如一只猫猫在玩球，标签只有猫；2.物理位置上相近的输出在视觉上不相近；3.做aug的时候裁剪的区域很有可能不是或不能体现目前的类别。本文针对的就是第一个问题，使用一个已经训好了的模型给出标签，这样的标签可能在多个类别中都有大与0的预测输出，然后再训后续的网络，使用KL散度让两个分布接近。
* 本文中使用KL散度作为loss函数，来衡量两个分布的距离。
* 其中改变标签的做法在整个类别上做是不对的，对每一张图片来做才有用。

---
## Crowd Counting
### Single-Image Crowd Counting via Multi-Column Convolutional Neural Network, CVPR 2016
* 对于crowd不是直接预测cnn回归器预测人数，而是做heat map，一方面这样的数据可以变相的aug好多，另一方面，从heat map到多少人的变换规律非常清楚，直接数就行了。这样就不要把这个任务也丢给网络来增加难度了。另外提出了一个新的数据集shanghaitech
### Switching Convolutional Neural Network for Crowd Counting，CVPR 2017
* 用一个switch网络从多个尺度的网络里面选一个，两种评价好不好的方式，mse即均方误差，还有mae，均绝对误差。
### Crowd Counting via Adversarial Cross-Scale Consistency Pursuit, CVPR, 2018
* 有两个不错的novel的地方，第一是多尺度的预测，平均加起来之后显得很模糊，所以他就用gan来作为额外loss，描述这种说不上来的让feature map变得sharp的loss。
* 第二点是为了使网络真正能对尺度变化鲁棒，设计了一个loss，使得网络对于原图的输出，和先把图像划分成四块，再过网络，再合起来的结果尽量接近（L2 loss），在这个约束下让网络对尺度尽量鲁棒。

---
## Face Detection
### Scale Aware Face Detection, CVPR,2017
* 思路是，之前的工作想要在测试时防止尺度的检测影响的话，需要多个尺度检测再融合，但是这样计算量会很大。这篇论文使用一个小网络对尺度进行一个预先的预测，再调整到合适的尺度下用单尺度的fasterrcnn进行检测。

### Mixed Supervised Object Detection with Robust Objectness Transfer，PAMI，2018
* 讲了怎么样在有监督和弱监督混合的场景中做detection，训练objectness的信息。首先在全监督的数据集上训objectness，对于只有分类标签的数据，不能直接训练，这里和gan思想有点类似，训练一个分类器来判断输入是从哪一个domain来的，然后为了对抗，把这个地方传回去的梯度乘以-1，也就是让之前的部分训得分不出是哪个domain，就把两个不同分布的差异抑制了，这样训出来的objectness就可以广泛的使用。

---
## Face Landmark
### Face Alignment in Full Pose Range: A 3D Total Solution， PAMI，2018
* gimbal lock，指的是用欧拉角（yaw，pitch，roll）表示三维旋转时，可能两组不同的参数转完后是同一个结果。这等于说少了一个自由度，需要用四元数来做旋转

---
## Face Recognition
### SphereFace: Deep Hypersphere Embedding for Face Recognition，CVPR，2017
* 通过修改softmax中的角度加大margin，来获取更小的类内方差
* 在softmax中加上margin的操作，通过加大约束能在球面上把决策区域变成一个一个圆形，感觉收的很紧，听有道理的

### Not Afraid of the Dark: NIR-VIS Face Recognition via Cross-spectral Hallucination and Low-rank Embedding,CVPR,2017
* 对于红外图像，把它伪造成rgb图像，然后按照rgb的网络去做识别

### VGGFace2: A dataset for recognising faces across pose and age，arxiv，2017
* 建立了一个新的benchmark VGGface2,这个数据集的重点是涵盖了所选的人的不同的角度和年龄段。

### Large-scale Datasets: Faces with Partial Occlusions and Pose Variations in the Wild，arXiv，2017
* 是一个大pose和遮挡的脸部数据集

### Longitudinal Study of Child Face Recognition， arXiv，2017
* 跟踪了一些小孩从2-18岁的图片来探究随年龄变化后人脸识别的性能，图片数量上还是有限，但是对于拐卖，寻人这样的问题意义比较大。

### Reconstruction-Based Disentanglement for Pose-invariant Face Recognition，ICCV，2017
* 把两张不同pose的人脸，对feature进行disentangle，一部分只保留identity信息，一部分保留pose信息，然后自己的两段特征，自己和别人的拼接特征再计算重建loss，用这种方式最后保留下的特征就可以不受pose影响了，这个idea（如果原创）的话还是很有启发意义的。
* 用重建而不是feature matching，可以避免约束过强

### Pose-Robust Face Recognition via Deep Residual Equivariant Mapping，CVPR，2018
* sensetime的论文，主要是为了解决大pose的问题，从原始特征后面接一个变换输出新特征再加权求和，中间用系数控制正脸和侧脸的比例，相当于把正脸的feature和侧脸的加起来，使用比例松弛一下，这样更接近在同个空间中。

### NormFace：l2 Hypersphere Embedding for Face Verification，ACM MM 2017
* 提到了几个很有意思的观点：1.在没有bias的时候，softmax会倾向于让特征的幅度变大来更容易的完成分类。如果x的分类能力是a，wx的分类能力是b的话，b就会比a明显好，但是这个scale是w提供的，x本身可能并没有变强，而人脸识别的特殊性在于比对最后用的还是x，归一化后可以抑制这种现象。
* 建议对softmax中的w，和最后的特征都做norm，但是建议norm到一个更大的值而不是norm到1（比如10）.
* norm的时候是x/sqrt(x^2+e)，这个e是为了防止归一化崩。

---
## Fine-grained Recognition
### Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition，CVPR，2017，oral
* 提到一个比较有意思的地方是在网络里面如果需要crop的话不是连续操作无法求导，一个近似的办法是用一个软的mask去乘，这个mask可以用两个sigmoid的分段组合来做。

---
## Fool Network
### Universal adversarial perturbations， CVPR，2017
* 用很简单的算法计算扰动，是global的，叠加在所有图像上能够使得很大比例的识别出错

---
## GAN
### Generative Adversarial Nets，NIPS，2014
* GAN，生成对抗网络，参考代码位于：https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
* idea是训练两个网络，一个是generator，用来生成图片愚弄discriminator，一个是discriminator，用来分辨图像是生成的还是真实的，分两步训练，两者互相对抗。disciminator是一个普通的CNN网络，而generator的输入是噪声，主要由卷积和上采样组成。
* 细节上，generator后面接上discriminator组成大的网络。训练discriminator的时候输入是真实图片和生成图片一起，标签是1（真实）和0（生成）并在一起。对于generator，训练的时候输入是噪声分布，标签全都是1，二者的损失函数都是binarycrossentropy。

### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks，ICCV，2017
* 核心思想是gan是一个可以不用pair的style transfer工具，那么既然能从a到b，也能从b到a，这样就结成了环，分为两部分loss，一部分是普通的gan的loss，另一部分是a到b，再b到a这样其实是重建的过程，加一个重建误差。

---
## Human-Object-Interaction Detection(HOI)
### Detecting and Recognizing Human-object Interactions,arXiv,2017
* 这个问题的目的是通过图像，得到<human，verb，object>的三元关系在此之前提出这个任务的论文是：Visual Semantic Role Labeling ，computer science，2015

---
## Low-level Image Processing
### Distort-and-Recover：Color Enhancement using Deep Reinforcement Learning，CVPR，2018
* 用强化学习的思路来做color的enhancement，用当前图像和目标图像的差的t+1和t的差距作为reward，规定每一步可以执行的action（在一个大的离散集合里面挑），然后达到最优的结果，即哪种action都不会有正的reward。

---
## Network Compress
### ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression， ICCV，2017
* 核心思想是在i层对filter作出裁剪，那么使得i+1层的输出不会有太大变化，说明这样的filter是可以移除的，因为减少了channel数目，所以叫做thinnet

### DEEP GRADIENT COMPRESSION: REDUCING THE COMMUNICATION BANDWIDTH FOR DISTRIBUTED TRAINING,ICLR, 2018 submission
* 大意是为了加速多个节点的训练（比如数据并行的多机），高过某个阈值的梯度直接发，小的先积累起来，等一下再发。

---
## Object Detection
### R-FCN，NIPS，2016
* 速度上比faster-rcnn快了很多，最多是后者的20倍。
* rfcn中，每一个网格位置单独设定了一个通道，而不是每一类物体一个通道的原因是，做的是检测任务，并没有像素级的标注。所以只能按照位置来粗略地分一下。

### YOLO9000：better，faster，stronger，CVPR，2017
* 提高输入图片的分辨率是比较关键的步骤，比如把输入尺寸从224改到448，当然这是以后面的全都是卷积形式为保证的。

### Mask RCNN，ICCV 2017
* 相比较于faster-RCNN，加了一支预测物体mask的分支
* 这里为了得到mask，将ROIpooling改为了ROIAlign，具体含义是比如原图上160x160的像素块，在conv5的feature map上是10x10（16倍下采样），在fast RCNN里需要变成7x7再继续接全连接。那么就是强行对10x10打上7x7的网格，然后有的格点包住一个点，有的包住两个，再max一下。现在做法改为插值的，即从目标点周围按照距离加权来差值。这对于小目标是非常重要的

### Perceptual Generative Adversarial Networks for Small Object Detection，CVPR，2017
* 用gan来做超分辨，用对抗的思想使得超分辨越来越真实，另外还有一个实践经验就是用到了一个分类的分支网络，这个分支往往能够加速网络收敛

### Focal Loss for Dense Object Detection，ICCV，2017
* 重新思考了entropy loss，发现它对容易分的样本挖掘不足，于是提出了focal loss，使得在一个batch中，容易分得样本对于loss比例减小，加大难分的样本的loss，再coco上面，对比其他single shot的分类器效果最好。

### Weakly Supervised Cascaded Convolutional Networks，CVPR，2017
* 利用弱监督做detection，网络结构和很多强监督的如maskrcnn等等很像，但是这么做还能涨点，还挺大胆的。

### DSOD: Learning Deeply Supervised Object Detectors from Scratch，ICCV，2017
* 看起来没有过拟合的原因似乎主要是由于用的densenet本身参数量不多造成的？所以才可以不经过imagenet直接训coco和voc

### MegDet: A Large Mini-Batch Object Detector，submitted to CVPR2018
* coco2017冠军。在检测中用多机来加大batchsize是挺重要的，因为之前都太小了。另一点就是适应地做好多机上的BN

---
## Object Segmentation
### Large Kernel Matters —— Improve Semantic Segmentation by Global Convolutional Network,CVPR,2017
分析了分类和分割的不同，指出对于分割来说获取全局信息很关键。因此在后面的feature map上用1xk和kx1的卷积来作为kxk的去搞全图的卷积，提升了分割性能，有novelty，不错的文章

### Boundary-sensitive Network for Portrait Segmentation，arXiv，2018
* google做人像分割的，有挺多insight，在监督前景背景的时候，可以监督那些区域是边缘，这样，网络自己可能可以学着去学习遇到边缘了应该怎么处理，第二是加上属性的分类器，把额外的信息传回去，让网络能够自己学习遇到长头发短头发该怎么处理等等。

### Macro-Micro Adversarial Network for Human Parsing，ECCV，2018
* 用GAN来保持global的语义一致性和local的纹理一致性，点有两个，第一是分到global和local这两个层面都有要做的事情，另一个还是那一点，当他要保证各种一致性而这种一致性又无法用数学公式写出来的时候，就用一个gan来做。

---
## OCR
### Detecting Text in Natural Scenes with Stroke Width Transform，CVPR，2010
* SWT，非常natural的方案，从笔画入手，先做canny，边缘的每一个像素点都看作是笔画的candidate，然后求梯度往法线方向走，到达另一个边缘，作为他的笔画宽度，然后相邻的像素就看笔画宽度近不近，用这种方法得到字母，然后得到line，作为text检测结果。

### Deep Features for Text Spotting，ECCV，2014
* 用CNN特征来做文字区域的检测和文字的识别。

### Deep Direct Regression for Multi-Oriented Scene Text Detection，ICCV，2017
* 检测的时候对于字体有一个技巧，就是在难说是不是字的地方设为“not care”区域，然后这部分训练的时候会忽略掉
* 为了检测旋转的字体，回归的时候输出的是四个端点，这样旋转角度就包含在里面了。然后回归也不像fasterrcnn那样，而是类似densebox，每个点都进行预测。

### Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework，ICCV，2017
* endtoend的文字区域检测和识别，在网络检测的时候预测box包含了旋转角度，然后对特征插值得到一个转正的特征，再经过fcn到一个固定尺寸的特征取识别，后面这步特征的旋转和插值，还有固定尺寸特征识别很机智。

---
## Pose Estimation
### Stacked Hourglass Networks for Human Pose Estimation， ECCV， 2016
* 这篇是没有pair-wise关系的工作中效果最好的一篇。
* 注意代码中hourglass是递归生成的，因此实际上最小的glass单元是图三最中间只有一道跨越的那一部分。实际上选取了四个单元来作为一个hourglass。每个单元分辨率缩小两倍，因此整体上是八倍。
* 在论文中hourglass单元不停地堆叠，当然有网络深化的作用。同时，pose estimation也可以是一个不断迭代推断的过程。这一步很可能是性能得到提升的关键，因为之前的工作顶多是卷积-解卷积这一套，加上不同层的feature的fusion，在这里是反复迭代的。用论文的话就是Repeated bottom-up, top-down inference with stacked hourglasses alleviatesthese concerns. Local and global cues are integrated within each hourglass module, and asking the network to produce early predictions requires it to have a high-level understanding of the image while only partway through the full network. Subsequent stages of bottom-up, top-down processing allow for a deeper reconsideration of these features

### DeeperCut: A Deeper, Stronger, and Faster  Multi-Person Pose Estimation Model，ECCV，2016
* 如果将vgg替换成resnet来做分割的话，那么后者的感受野可以更大一些，可以考虑扩大图像分辨率到340等等。
* 就pose estimation来说，任务不一样，用来评价的标准也不一样。在MPII上有两种任务，一种是单个人的，这种的评价是PCK，而multi-person的任务评价标准是AP。对于PCK，就是每个关节按照阈值去判断，如果和真值小与这个阈值就是正确的，否则是错误的。

### Convolutional Pose Machines，CVPR，2016
* 为什么需要卷积层不停堆叠以及中间加监督？原因一方面是因为深度深了一些，另一方面随着层数的深入，感受野是不断增加的，这样很多时候是更有理由推断出这里是什么的。
* 关于感受野的计算，需要从最后开始倒推，依据是上一层有多少个单元会影响下一层的输出。以vgg16为例，应该是pool5的每一个元（7x7中的每一个）对应于原来图像中的212x212，所以说把图像扩大到256x256还是会有意义的。

### Learning Feature Pyramids for Human Pose Estimation， ICCV，2017
* 新的stateoftheart，在hourglass的基础上加入了多尺度的金字塔，实现方式是分成不同尺度pooling，然后自己卷一卷，在upsampling上来加在一起，这个工作的实现多尺度比较科学，因为feature map的size不会增长。这样的话就可以多轮回几次。

### Human Pose Estimation using Global and Local Normalization，ICCV，2017
* 在出步检测到joint之后，加一个旋转环节，使得整个图像尽量按照头和脚是上下的，对于part也加上这样的环节

### Associative Embedding: End-to-End Learning for Joint Detection and Grouping，NIPS，2017
* 对于multi-person，怎么去建模呢，本文致力于在训练网络的时候就进行区分。对于两个人，让他们关节点的值不一样，比如第一个人heatmap上keypoint处值为1，第二个人为2。然后训的loss设计使得一个组减去1后很小，一个组减去2很小，两个组互相减就很大，而测试的过程其实也就是解这个方程使得loss最小的过程，这个结果比paf要高的不少，是目前最好的结果。

### RMPE: Regional Multi-Person Pose Estimation，ICCV, 2017
* 也是做bottom-up的pose，在检测框后，对单人前后加上STN的SDTN来转正再做pose estimation

### End-to-end Recovery of Human Shape and Pose，arxiv，2017
* end2end的可以估计2D和3D的pose以及分割的shape，有一个有趣的地方是怎么判定3D的pose那个模型估计的怎么样，即怎么设计loss，这里对这种不好描述的loss都用一个GAN来处理，也是很好的启发。

---
## Semi-Supervision
### Learning by Association A versatile semi-supervised training method for neural networks， CVPR, 2017
* 试图让网络学习好的embedding，准则是用一个walker，从同一个类别走出去，在走回来还是这个类就认为是好的类。

---
## Solver
### ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION，ICLR，2015
* adam是adptive moment estimation，自适应矩估计。对每个参数都有一阶矩和二阶矩的估计，可以防止某些时候的步长变化太剧烈而陷入局部最优中，实际性能是目前已知最好的梯度优化方法

### Cyclical Learning Rates for Training Neural Networks，arXiv，2015
* 提出了循环的调节lr的训练方法，三角波，理由是当已经慢慢陷入鞍点的时候，重新启用大的lr可能能快速跨越鞍点，很work，在很多数据和任务上都可以涨点

### Averaging Weights Leads to Wider Optima and Better Generalization,arXiv,2018
* 提出了一个新奇的观点，把几代存储模型的w平均起来当做初始值继续训练，效果有待验证

---
## Style Transfer
### Image Style Transfer Using Convolutional Neural Networks，CVPR，2016
* 描述如何进行style transfer。方法是输入真实图像，目标风格图像和一张噪声图像，对于真实图像，loss为每一层，每个点的特征l2loss，对于目标风格图像，则是统计值的接近，然后网络，真实，目标风格图像都不懂，只对噪声图像求导最小化，最后得到一张迁移了风格的图像。

---
## Tracking
### Detect to Track and Track to Detect，ICCV，2017
* 两个网络出t帧和t+tt帧的feature，中间做相关，然后监督bbox在tt时间段内的偏移量，测试的时候中间那部分不会用到，但是通过这样把它训得更好，在ImageNet的VID上结果最好

### VITAL：visual tracking via adversarial learning，CVPR，2018（spotlight）
* 用G生成各种mask的样本，让D也要能扛得住来提升算法性能.

---
## Video
### SegFlow: Joint Learning for Video Object Segmentation and Optical Flow，ICCV，2017
* 春姐的论文，在视频中用一个endtoend的模型同时利用光流和分割来在视频中分割物体

### Blazingly Fast Video Object Segmentation with Pixel-wise Metric Learning，CVPR，2018
* 第一点是把tripletloss用在分割上，当做是一个检索任务，这里猜测tripletloss需要大量的样本才能做，其他任务可能不这么多，但是video里面的像素是足够多的。
* 第二点是如何应用空间和时间的信息，他把时间和空间的坐标直接拼接在特征（这里叫做embedding）后面作为一种简单的方法。
* 直接使用triplet loss会有一些问题，由于同一个物体的不同位置本身就存在类间方差，强行学会学的不好，所以要求某一类中最小的那个距离小于和负样本中最小的那个的距离就行了。
* ablation study 模型简化测试。看看取消掉一些模块后性能有没有影响。

---
## Zero-shot Learning
### discriminative learning of latent features for zero-shot recognition，CVPR，2018
* 最早的zero shot做法是人为的手工定义一些特征，然后训练分类器，看看某些类别之间有没有这些属性，然后比较相似不相似。到后面直接比较有点简单了，变成了中间间隔一个矩阵过渡图像特征和属性特征。
* 这里把attribute分成两段，第一段是用户标注的属性向量，直接用wx和它的点积计算相似度然后让结果在c类上最大就行了。第二段是要去学习的，利用wx的后半段做一个tripletloss。这样这些向量离这些类别比较接近，并且不会完全和类别的特征重叠（因为经过了w）

