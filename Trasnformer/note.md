# understanding of transformer

<div align="center">
<img src=./note_figures/transformer.png width=60% />
</div>

本节主要根据`./code/AnnotatedTransformer.ipynb`进行学习理解，将Transformer重新实现组织到`./code/transformer.py  ./code/modules.py`中。    

## Positional Encoding

因为多头自注意力并没有考虑token在句子中的位置，token之间的先后关系
为了引入序列长度，token在序列中的位置信息，故在Embedding后引入一个Positional Encoding

下面给出计算公式:
  
$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
  
pos为token在句子中的位置，i代表emb_dim中的位置    
`2i`代表偶数位置，`2i+1`代表奇数位置    
偶数位置使用`sin`, 奇数位置使用`cos`    
将计算后得到的和Embedding相同维度的向量对应相加    
最后再经过dropout `(base model: p=0.1)`    

<img src=./note_figures/positional_emb.png width=80% />

##### 公式中，pos代表了token的绝对位置信息      
此外，通过三角函数公式可以得到：   
<img src=./note_figures/positional_emb_formula.png/>   

对`pos+k`位置而言，其向量表示可以理解为`pos`位置与`k`位置的向量的线性组合    
##### 其中蕴含了相对位置的信息

## Multi-Head Attention

### self-attention    
<img src=./note_figures/self_attention.png width = 70%/>
 
通过计算`q,k`的相似度（点乘），得到一个实数，越大越相似，从而该位置的权重就越大     

这种通过`query`和`key`的相似性程度来确定`value`的权重分布的方法被称为`scaled dot-product attention`。   

attention机制的理解：类比数据库系统的`qeury,key,value`，输出更关注`qeury`和`key`更相似地方的`value`   

通过这样的方法来抽取句子中token之间的关系   

### Multi-Head Self-attention      
<img src=./note_figures/multihead_self-attention.png width = 70%/>  

`Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions.`

### Masked Multi-Head Attention    
在预测生成阶段，Decoder的输入并不能看到一句完整的输入，而是第i个词的输出作为第i+1个词的输入    
故在训练的时候，不应该给Decoder输入句子每个位置的词都看到完整的序列信息，应该让第i个词看不到第j个词(j>i)     

### Cross Attention     
同样是Multi-Head Attention，但输入的q是经过一次Masked Multi-Head Attention 特征提取后输出    
k,v都来自Encoder的输入    

### 在Transformer中，所有注意力机制计算都是如下公式       
$$
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

假设 q 和 k 是独立的随机变量，平均值为 0，方差 1，这样他们的点积后形成的注意力矩阵为 $q⋅k=\sum_{i=1}^{d_k}{q_i k_i}$ ，均值为 0 但方差放大为 d_k 。为了抵消这种影响，我们用 $\sqrt{d_k} $来缩放点积，可以使得Softmax归一化时结果更稳定（不至于点积后得到注意力矩阵的值差别太大），以便反向传播时获取平衡的梯度

## Layer Norm & Batch Norm    

### Batch Normal      
<img src=./note_figures/batch_norm.png width = 50%/> 

对一个batch中的样本在同一维度进行normalization    
但这样的处理在序列模型中并不是很好      
一些直观的理解：①序列问题中为了进行并行化运算，需要对一个batch中的句子进行padding，故同一个batch中不是每个样本都有完全一样的特征空间；②对一个batch中每个样本同一个位置的token做normalization，这需要每个句子在同样位置包含一样的语义信息，似乎不是很有道理；故在同一batch中对每个样本某个特征取出来的平均值和标准差不一定有很好的效果。      

### Layer Norm         
<img src=./note_figures/layer_norm.png width = 30%/> 

对每一个单词的所有维度特征(hidden)进行normalization

一言以蔽之。BN是对batch的维度去做归一化，也就是针对不同样本的同一特征做操作。LN是对hidden的维度去做归一化，也就是针对单个样本的不同特征做操作。

## Residual network    
<img src=./note_figures/resnet.png width = 60%/>       

在深度网络的反向传播中，根据链式求导，会出现多个连乘，这样可能会导致一个小数越乘越小，越来越趋于0，出现梯度消失。     
而在transformer中，需要将多个Encoder，Decoder Block叠在一起，在这样深的网络中使用res_net可以很好的缓解梯度消失。    


## Feed forward      
两层全连接网络   

$$\mathrm{FFN}(X) = \mathrm({ReLU}(XW_1+b_1))W_2+b_2$$
     
其中，X的特征维度默认为512,中间隐藏层的特征维度默认为2048，输出的特征维度为512     
从而可以增加一些非线性变化   

## 模型训练  
### 初始化参数   
 
使用Xavier Initialization


### Optimizer   

#### Adam optimizer    
with $\beta_1=0.9$ , $\beta_2=0.98$ and $\epsilon=10^{-9}$ .   
同时使用下述公式控制lr变化    

$$
lrate = d_{\text{model}}^{-0.5} \cdot
  \min({step\_num}^{-0.5},
    {step\_num} \cdot {warmup\_steps}^{-1.5})
$$

先让`lr`线性上升，`step_num == warmup_steps`后呈-0.5的指数下降

取`warmup_steps = 4000`    
<img src=./note_figures/warmup.png width = 80%/> 

