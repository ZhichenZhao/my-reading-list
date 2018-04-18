# Waterbro's Reading List on Computer Vision
这份笔记分为每种topic以及topic下的论文，topic按照字典序排列，论文按照时间顺序排列。

## base model
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
* 是对resnet的一种阐述和改进，只不过占用空间是个问题。在cifar上是最好的结果。看起来道理挺到位的，不知为何没有在更大的地方work

---
## Crowd Counting
### Single-Image Crowd Counting via Multi-Column Convolutional Neural Network, CVPR 2016
* 对于crowd不是直接预测cnn回归器预测人数，而是做heat map，一方面这样的数据可以变相的aug好多，另一方面，从heat map到多少人的变换规律非常清楚，直接数就行了。这样就不要把这个任务也丢给网络来增加难度了。另外提出了一个新的数据集shanghaitech
### Switching Convolutional Neural Network for Crowd Counting，CVPR 2017
* 用一个switch网络从多个尺度的网络里面选一个，两种评价好不好的方式，mse即均方误差，还有mae，均绝对误差。

---
## Face Detection
### Scale Aware Face Detection, CVPR,2017
* 思路是，之前的工作想要在测试时防止尺度的检测影响的话，需要多个尺度检测再融合，但是这样计算量会很大。这篇论文使用一个小网络对尺度进行一个预先的预测，再调整到合适的尺度下用单尺度的fasterrcnn进行检测。

### Mixed Supervised Object Detection with Robust Objectness Transfer，PAMI，2018
* 讲了怎么样在有监督和弱监督混合的场景中做detection，训练objectness的信息。首先在全监督的数据集上训objectness，对于只有分类标签的数据，不能直接训练，这里和gan思想有点类似，训练一个分类器来判断输入是从哪一个domain来的，然后为了对抗，把这个地方传回去的梯度乘以-1，也就是让之前的部分训得分不出是哪个domain，就把两个不同分布的差异抑制了，这样训出来的objectness就可以广泛的使用。

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




