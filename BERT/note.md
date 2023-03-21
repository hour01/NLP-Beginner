# BERT

BERT（Bidirectional Encoder Representations from Transformers） 是一个语言表示模型(language representation model)。它的主要模型结构是trasnformer的encoder堆叠而成，它其实是一个2阶段的框架，分别是pretraining，以及在各个具体任务上进行finetuning。

## 结构
BERT主要由三部分组成：输入部分、中间部分、输出部分
### 中间部分
下图所示的就是BERT中间部分的结构，由多个`Transformer encoder`叠在一起，内部结构和Transformer一致
<div align="center">
<img src=./note_fig/bert.png width=70% />
</div>

### 输入部分
<div align="center">
<img src=./note_fig/emb.png width=70% />
</div>

为了使得BERT模型适应下游的任务（比如说分类任务，以及句子关系QA的任务），输入将被改造成[CLS]+句子A（+[SEP]+句子B+[SEP]） 其中

- [CLS]：代表的是分类任务的特殊token，它的输出就是模型的pooler output
- [SEP]：分隔符
- 句子A以及句子B是模型的输入文本，其中句子B可以为空，则输入变为[CLS]+句子A

在BERT中，输入的向量是由三种不同的embedding**求和**而成，分别是：

`wordpiece embedding`：单词本身的向量表示。WordPiece是指将单词划分成一组有限的公共子词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡。

`position embedding`：将单词的位置信息编码成特征向量。BERT不同于Transformer中使用一个固定的函数式的position embedding，而是是初始化一个position embedding，然后通过训练将其学出来。

`segment embedding`：用于区分两个句子的向量表示。例如`[CLS], A_1, A_2, A_3,[SEP], B_1, B_2, B_3, [SEP] 对应的segment的输入是[0,0,0,0,1,1,1,1]`

#### BPE & WordPiece
>1.实际应用中，模型预测的词汇是开放的，对于未在词表中出现的词(Out Of Vocabulary, OOV)，模型将无法处理及生成；
2.词表中的低频词/稀疏词在模型训练过程中无法得到充分训练，进而模型不能充分理解这些词的语义；
3.一个单词因为不同的形态会产生不同的词，如由"look"衍生出的"looks", "looking", "looked"，显然这些词具有相近的意思，但是在词表中这些词会被当作不同的词处理，一方面增加了训练冗余，另一方面也造成了大词汇量问题。

此类方法主要就是为了解决上述问题，从而**将原来token从一个英文单词变为更细粒度的表示**，一个是字符级别，一个是subword，后者的粒度介于单词和字符之间，正如上述第三点所述。

##### BPE（Byte-Pair Encoding）
BPE获得Subword的步骤如下：

1.准备足够大的训练语料，并确定期望的Subword词表大小；
2.将单词拆分为成最小单元。比如英文中26个字母加上各种符号，这些作为初始词表；
3.在语料上统计单词内相邻单元对的频数，选取频数最高的单元对合并成新的Subword单元；
4.重复第3步直到达到第1步设定的Subword词表大小或下一个最高频数为1.

下面以一个例子来说明。假设有语料集经过统计后表示为{'low':5,'lower':2,'newest':6,'widest':3}，其中数字代表的是对应单词在语料中的频数。

拆分单词成最小单元，并且在每个单词后加入</w>用来判断单词边界，并初始化词表。这里，最小单元为字符，因而，可得到
<img src=./note_fig/bpe_1.png width=30% />

在语料上统计相邻单元的频数。这里，最高频连续子词对"e"和"s"出现了6+3=9次，将其合并成"es"，有
<img src=./note_fig/bpe_2.png width=30% />
由于语料中不存在's'子词了，因此将其从词表中删除。同时加入新的子词'es'。一增一减，词表大小保持不变。

继续统计相邻子词的频数。此时，最高频连续子词对"es"和"t"出现了6+3=9次, 将其合并成"est"，有
<img src=./note_fig/bpe_3.png width=30% />
如此迭代执行

在得到Subword词表后，针对每一个单词，我们可以采用如下的方式来进行编码：

1.将词典中的所有子词按照长度由大到小进行排序；
2.对于单词w，依次遍历排好序的词典。查看当前子词是否是该单词的子字符串，如果是，则输出当前子词，并对剩余单词字符串继续匹配。
3.如果遍历完字典后，仍然有子字符串没有匹配，则将剩余字符串替换为特殊符号输出，如”<unk>”。
4.单词的表示即为上述所有输出子词。
解码过程比较简单，如果相邻子词间没有中止符，则将两子词直接拼接，否则两子词之间添加分隔符。

##### WordPiece(used in BERT)
与BPE算法类似，WordPiece算法也是每次从词表中选出两个子词合并成新的子词。与BPE的最大区别在于，如何选择两个子词进行合并：BPE选择频数最高的相邻子词合并，而WordPiece**选择能够最大程度提升语言模型概率的相邻子词加入词表**。

假设句子$S=(w_1,w_2,...,2_n)$由n个子词组成，$w_i$表示子词，且假设**各个子词之间是独立存在的**，则句子的语言模型似然值等价于所有子词概率的乘积：

$logP(s)=\prod_{i=1}^n logP(w_i) $

假设把相邻位置的x和y两个子词进行合并，合并后产生的子词记为z，此时句子S似然值的变化可表示为：
$logP(t_z)-(logP(t_x)+logP(t_y))=log(\frac{P(t_z)}{P(t_x)P(t_y)}) $

每次选择两个能使得上式取值最大的两个字词合并

### 输出部分

主要考虑对接其下游任务，主要有两内输出：
- pooler output：对应的是[CLS]的输出。
- sequence output：对应的是所有其他的输入字的最后输出。


## Pre-training
语言预训练模型的主要是采用大量的训练预料，然后作无监督学习。

- BERT模型的目标： 传统的语言模型就是预测下一个词，例如我们键盘的输入法。一般采用的是从左到右的顺序。但是这就限制了模型的能力，不能考虑到后面的序列的信息。如果要考虑双向的信息的话，可以再使用从右到左的顺序，预测下一个词，然后再将两个模型融合，这样子就考虑到了双向的信息（ELMO的做法）。但是这样做的坏处主要是:
1.参数量变成了之前单向的两倍，也是直接考虑双向的两倍
2.而且这个对于某些任务，例如QA任务不合理，因为我们不能够看完答案再考虑问题
3.这比直接考虑双向模型更差，因为双向模型能够在同一个layer中直接考虑左边和右边的context

### task-1: Masked Language Model(MLM)
随机从输入句子中mask几个词，然后预测mask的这个词，类似于让模型做完形填空
上述的mask不是如同Transformer中使用一个mask矩阵，而是引入一个<Mask>token，对输入语句的15%做：
- 80%的情况是替换成[MASK]
- 10%的情况是替换为随机的token
- 10%的情况是保持不变
### task-2: Next sentence prediction(NSP)
很多需要解决的NLP tasks依赖于句子间的关系，例如问答任务等，这个关系语言模型是获取不到的，因此将下一句话预测作为了第二个预训练任务。该任务的训练语料是两句话，来预测第二句话是否是第一句话的下一句话。
- 每次选取两个句子(A,B)作为训练样本，其中50%的时候B是A的下一个句子，其余时候B为从语料库中随机选择的句子。 

## Fine-tuning

预训练好模型后，可以根据具体任务直接取上述输出部分的输出或者在此基础上增加一层或几层神经网络实现相关任务。
<div align="center">
<img src=./note_fig/fine_tuning.png width=70% />
</div>

左上和右上是对于分类任务，右下是sequence to sequence模型
左下是根据输入的文章和问题，得到问题在文中的位置(输出起始和终止)

## Whole Word Masking Models
Whole Word Masking (wwm)是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。

原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。这样没有考虑一个完整的词才代表一个较独立的语义。 
在Whole Word Masking (wwm)中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask。

对于中文的输入也有对应的解决
[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf)
<div align="center">
<img src=./note_fig/wwm.png width=90% />
</div>

## 使用
BERT 模型的使用主要有两种用途：
- 当作文本特征提取的工具，类似Word2vec模型一样
- 作为一个可训练的层，后面可接入客制化的网络，做迁移学习

例子如`./code/example`所示
相关链接：
[huggingface,预训练模型下载](https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.)
源码：https://github.com/google-research/bert