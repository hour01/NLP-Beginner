# Text classification

## Dataset
sst-2(GLUE)
| train | dev | test |  
| ----- | --- | ---- |  
| 67349 | 872 | 1821 |  
Stanford Sentiment Treebank (SST) 包含了11,855句电影评论，原始数据中被划分为了五类，实验采用二分类，故丢弃其中性评价，使用剩下作为数据集；
同时其原始数据集不止包含单句的标注，还有短语级的类别标注，将短语级别类别标注加入训练集，可以得到上表所示的训练集大小(GLUE)，但测试集并没有公开标签且存在部分句子不属于sst原始数据集。
自行对原始数据集按照数据集介绍切分后得到的的数据如下
| train | dev | test |  
| ----- | --- | ---- |  
| 6568  | 825 | 1750 | 
本次实验采用sst-2(GLUE) 67349/872/1821


## CNN&LSTM
使用与预训练词向量GoogleNews作为Embedding层（300维）
### CNN
主要参考[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882v2.pdf) 
使用卷积核来提取输入句子的特征，其中卷积核大小为(n,embedding_size)，n作为一次卷积计算包括的单词。
对于某一种卷积核，输入的句子向量矩阵通过该层后得到形如(n_filters,sentence_len - n + 1)的输出 
此时每个通道的信息为卷积核对输入句子按顺序进行卷积计算后得到的序列，将矩阵表示的句子压缩为一维长度表示
- 此过程可以类比N-gram语言模型，考虑了相邻几个词组成的词组含有很好的语义信息，故卷积核一次对相邻的几个词组成的嵌入矩阵进行卷积计算，提取特征  
- 在前面使用最大熵模型进行文本分类时，将词数字化采用高维one-hot表示，n-gram表示考虑到one-hot中每个1不止代表一个词的出现，还可以代表一个词组（相邻）几个单词的连续出现，借助这样的思想，在神经网络中，使用CNN的卷积核固定其中一个维度和embedding维度相同，可以指定对相邻区间内的单词进行小范围内的特征抽取 
通过在每个通道上进行max_pooling，选取最大值，进一步压缩提取信息，最后把所有卷积核通过上述操作得到的结果拼接起来作为cnn提取到的分类特征。 


### LSTM
上述使用CNN抽取序列模型特征有一个缺点
max_pooling对整个通道中所有数据进行操作，最终获得到一个最能代表整个序列的特征，但同时也丢失了序列的位置信息 
而LSTM按顺序对每个词向量进行计算，天然具有识别抽取位置信息的能力 
本次实验使用LSTM最后一个隐藏单元的值作为LSTM对整个句子情感信息的表示 
- 同时也尝试将LSTM每个时刻的输出(len, output_size)加上对len的max_pooling作为情感信息表示，但实验效果和上述处理差别不大 
- 同时使用bi-LSTM，为隐藏单元增加反向信息，做实验后发现在这个该分类任务上相比单向LSTM最终结果没有取得很明显的增益，但通过观察loss的变化，可以发现bi-LSTM收敛的更快（更少的epoch）
- 增加bi-LSTM的层数，明显观察到收敛速度的加快，同时也导致过拟合比单层的更严重，考虑将bi-LSTM隐藏单元维度减半，观察到有助于缓解过拟合
同时调参的过程中发现使用小batch对该任务有很大提升，`batch_size=16` 

## (bert/elmo)-(bilstm/cnn)
通过引入预训练好的bert（bert-base-uncased）或elmo（固定参数），直接作为bi-lstm，cnn的embedding层，获取句子的embedding表示 
其中bert表示的词向量维度为768，elmo表示的维度为1024
实验观察到相比使用预训练词向量GoogleNews模型表现有质的提升  



## 结果(GLUE benchmark)
|model|Acc|
|-----|---|
|CNN  |83.0|
|BiLSTM  |83.1|
|LSTM    |81.7|
|Bert-BiLSTM |89.8|
|Bert-CNN |88.0|
|Elmo-Bilstm|88.0|
|Elmo-CNN|88.9|


## SST-2文本分类总结 
根据[papers with code](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary) 
目前在该数据集上的SOTA模型是`SMART-RoBERTa Large`，同时观察到榜单上准确率排名靠前的方法大多基于预训练模型，RoBERTa,DeBERTa,XLNet等等。
在Bert的基础上，通过不断改进预训练任务，增加预训练时间，增加训练集数量。

Bert在预训练方法上有一些设计的问题：
- bert的MLM引入的`[mask]`token在下游任务中并不存在，会导致`pretrain-finetune discrepancy` 
- bert在与其预训练方法不大相似的下游任务中，表现出来得效果没有像那些与预训练任务相似的下游任务那么优异，比如与MLM和NSP任务距离比较远的生成、长文本依赖建模、结构化知识抽取等任务没有取得同等惊艳的效果 

### XLNet
引入Permutation Language Model，结合了以GPT、ELMo为代表的AutoRegressive LM 和 以Bert为代表的Autoencoding LM 各自的优点：使得预训练过程既能像AE一样**同时利用上下文信息**，又能像AR一样**使得预训练阶段和下游任务一致，没有引入<mask>，符合大多数生成模型。** 
- PLM: 通过随机取一句话每个token进行排列组合的一种，然后将末尾一定量的词给 “遮掩”掉（和 BERT 里的直接替换为[mask] 不同，采用mask矩阵来处理），最后用 AR 的方式来按照这种排列方式预测被 “遮掩” 掉的词，这样模型就能通过**AR的单向方式来学习到双向信息**。

### RoBERTa 
- **采用动态masking**：相比Bert在数据预处理阶段就完成masking操作，RoBERTa是在训练句子输入模型前进行masking，以保证每轮masking都不同 
- **去掉NSP任务**：文中通过实验给出，每次输入多个连续句子直到最大长度512（跨越或不跨越文档，后者更优），这样的方式在句子关系任务上效果优于NSP。 
- **Training with Large Batch**: 更多数据、更多算力、更大batch 
- **Text Encoding**: 从Bert采用的 character-level BPE(3w词表)，转为byte-level BPE(5w词表)，细粒度词表一定程度解决了OOV问题。 

### DeBERTa 
- **Disentangled attention**: 每个token使用两个向量表示，分别对其内容和位置进行编码，而token之间的attention权重也是根据内容与相对位置使用disentangled matrices进行计算。例子：deep和learning之间的依赖性在相邻时比在不同的句子中出现时要强得多。 
- **Enhanced mask decoder**: 在预训练时用一个两层的Transformer decoder和softmax作为Decoder，并且在decoder的softmax之前将单词的绝对位置嵌入。例子： “a new store opened beside the new mall”，虽然store和mall的局部上下文相似，但它们在句子中起着不同的句法作用，句子中store作为主语，这种不同很大程度上取决于绝对位置。 
- **Virtual Adversarial Training**: 用于改进模型的泛化能力。 

### SMART-RoBERTa 
[SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/pdf/1911.03437v5.pdf)
针对现有**微调方法**中存在的在下游任务中 预训练模型的extremely high complexity 以及 aggressive fine-tuning 问题，提出了两种微调方法： 
**SMoothness-inducing Adversarial Regularization**
加入一个正则项:$min_\theta F(\theta)=L(\theta) + \lambda_s R_s(\theta) $
L是损失函数，后者有两种形式：
- $[A]:R_s(\theta)=\frac{1}{n}\sum_{i=1}^n max_{\|x_i -\bar{x_i}\|_p \leq \epsilon} l_s(f(x_i;\theta),f(\widetilde{x_i};\theta))$
- $[B]:R_s(\theta)=\frac{1}{n}\sum_{i=1}^n max_{\|x_i -\bar{x_i}\|_p \leq \epsilon} l_s(y_i,f(\widetilde{x_i};\theta))$
可以使得$f(x_i;\theta)$更平滑 

**Bregman proximal poinT optimization** 
在优化上式的时候，每一步迭代都加入一个正则化项，用来保存一定量之前学到的东西，不至于一下子调过头了。 
$\theta_{t+1} = arg min_\theta F(\theta) + \mu D_{Breg}(\theta,\theta_t)  $ 
$D_{Breg}(\theta,\theta_t)=\frac{1}{n}\sum_{i=1}^n l_s (f(x_i;\theta),f(x_i;\theta_t)) $ 

使用这样的方法在Bert、RoBERTa模型上，结合MT-DNN（多任务学习）取得了SOTA。
