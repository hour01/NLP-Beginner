# Frame-Semantic Parsing

## 任务目标    
识别出文本句子中被某些词语激发的语义框架（包括目标词、参数），而一个语义框架代表了某种语义信息，例如事件、关系等。以下是一个FrameNet中的例子。
<div align="center">
<img src=./note_fig/example_frame_net.png width=70% />
</div>

其中不同颜色表示不同的语义框架，以红色的PERFORMERS_AND_ROLES为例说明:`played`作为一个`target word`激发了一个`表演者和角色`语义框架，同时该语义框架下定义了若干语义参数，`Performer,Role,Performance`，每个参数都是该语义框架的一个语义成分共同组成了一个语义信息。

一般来说，Frame-Semantic Parsing 任务被分解为三个独立子任务：   
- **Target Identification**：识别出可以激发一个语义框架的词或短语，被称为目标词(target word)
- **Frame Identification**：识别出被激发的语义框架，即多分类问题
- **Argument Identification**：识别出定义在该语义框架下所有语义角色（arguments）

本次实验中，上述语义框架的识别采用[FrameNet Project](https://framenet.icsi.berkeley.edu/fndrupal/)中的定义。  

## 数据集介绍    
本次实验使用了`FrameNet version 1.7`，同时将数据组织为与[CONLL 2009](https://ufal.mff.cuni.cz/conll2009-st/task-description.html)类似的格式，采用BIO标注。   
处理后的数据集位于`./data/processed_data`下，其在单词层面上对句子中每一个单词进行了标注，样本之间用空行分隔，每一行是一个单词的所有标注，包括单词的`词性、原型、单词所属的语义角色(BIO tag)`以及该句子中的`target_word(lexical_unit.pos)`以及对应的`语义框架类别`。    
 
## Methods     
参考[Frame-Semantic Parsing with Softmax-Margin Segmental RNNs and a Syntactic Scaffold](https://arxiv.org/abs/1706.09528)，使用`pytorch`重新实现了文中的方法。（官方代码使用dynet框架）
> 对于TI和FI任务，实验结果可以达到文中的效果；对于AI，因为该方法运算量过大（没有使用batch，且个人使用的GPU需要10+小时运行一个epoch），故没有做出最终实验结果。
 
### Target Identification    
该部分的任务是对于一个输入的句子，给出该句子哪些词可以作为`Target word`。
#### Method     
可以将该问题看作一个二分类问题，分类对象是该句子中的每个词，给出每个词是否是`Target word`的判断。   
采用的具体方法：    
- 将输入的词语序列、对应的词性序列使用一个需要训练的词嵌入层进行向量化，得到的向量与每个词的Glove预训练词向量在词维度进行拼接。
- 输入BiLSTM提取每个时刻的输出做二分类。
- 使用交叉熵得到loss进行反向传播训练优化参数。

$y^{pred}=Linear(BiLSTM(v_1,...,v_n))$   

### Frame Identification   
给出输入句子，句子中的target word，该任务需要识别出该句子中的target word激发的语义框架类型。

#### Method    
直观来看，该问题就是一个多分类问题，判断一个target word在上下文句子中激发的语义框架类型。  

- 使用词嵌入层获取句子中每个词的表示，包括词本身、词性，输入BiLSTM提取每个词在句子层面的语义表示
- 从上述BiLSTM的输出中选取target word中每个单词对应时刻向量，再通过一个BiLSTM作为该target word的上下文信息表示，同时拼接上该target word的`lexical unit(词法单元)`和`词性`的嵌入向量作为target word的编码信息，接上全连接层输出每种语义框架的概率
- 通过上述输出概率计算loss时，首先提取该训练样本中`lexical unit`可能对应的所有语义框架 $f$ ，只选取出现在 $f$ 中的类别对应的概率求交叉熵损失进行参数优化。

$h^{token}=BiLSTM^{token}(v_1,...,v_n)$   
$h^{target}=LSTM^{target}(h_{lu_1},h_{lu_2},..,h_{lu_k})$，k是target word单词数     
$y_{pred}=Linear([h^{target};v_{lu};v_{pos}])$

### Argument Identification (Semantic Role Labeling)     
给定句子，句子中的target word，以及该target word所激发的语义框架，该任务需要在给定的句子中识别出该语义框架的各个语义成分(frame elements)。

#### Method     

每种语义框架包含的语义成分都是固定的，可以将该问题转化为对所有可能的span，预测其所属语义成分的多分类问题。 
> 由于数据集中直接对每个单词采用了BIO标注，也可以将该任务看作一个NER任务。 

##### Input Span Embeddings —— [Segmental RNN](https://arxiv.org/abs/1511.06018)           
每一个语义框架包含若干语义成分，且语义成分多以span的形式出现，为了能够对各个span进行识别，需要先对句子中每个可能的span进行向量化表示。使用两个BiLSTM，其中一个用于给出整个句子基本的向量表示 $h^{token}$ ，在此基础上使用另一个BiLSTM对需要表示的 $span:(i,...,j)$ 进一步提取特征得到 $h^{span}$ ，连接正向、反向在该span内最后一个时刻的输出向量作为该span编码向量的最终表示 $h_{i:j}$ 。

$h^{token}=BiLSTM^{token}(v_1,...,v_n)$   
$h^{span}=BiLSTM^{span}(h_i^{token},...,h_j^{token}) $   
$h_{i:j} = [\overrightarrow{h^{span}_j};\overleftarrow{h^{span}_i}]$   

对于上述得到的 $h_{i:j}$ ，还需要融合在`Target Identification` 和 `Frame Identification` 得到的信息，即加入`target word`和`语义框架类别`的嵌入向量 $v_{frame}$ ；同时加入一些span在句子中的信息，即span的长度和这个span在句子中相对于target word的相对位置，二者分别做embedding拼接后表示为 $μ_{span}$ 。最终拼接上述所有向量得到span的最终表示，其中对span最大长度做了限制。      
$v_{span}=[h_{i:j};v_{frame};μ_{span}]$

##### Segment Scores      
为了能够预测每个span可能属于的语义成分，需要在span和可能的语义成分之间进行组合求概率(score)。其中，因为前面已经对语义框架类别进行了预测，一种语义框架下所对应可能的语义成分是固定的，所以span只需要和对应语义框架下可能的语义成分进行组合求得分概率即可。其中语义成分类别记作 $y$ ，将其做向量化嵌入后表示为 $v_y$ ，同时每一个待选span可以表示为 $s=<i,j,y>$ ，最终每种组合的score表示为：   
$score(s) = W_2·ReLU(W_1[v_{span};v_y])$     
<div align="center">
<img src=./note_fig/ai_arch.png width=40% />
</div>

##### [Softmax-margin ](https://aclanthology.org/N10-1112/)   

上述过程对长度为l的句子进行了逐个枚举得到了 $(l+1)*l/2$ 种span，加上与语义成分类别的组合最终会得到 $O(l^2*class_{number})$ 种得分结果，但其中大部分都不属于任何语义成分，故采用softmax-margin求loss，该方法会使得模型预测更不容易出现`False negtive`和`False positive`。     

$$loss=-log\frac{exp(score(s^*))}{Z}$$      

$$Z=\sum_sexp{score(s)+cost(s,s^*)}$$   

$$cost(s,s^*)=\alpha FN(s,s^*)+FP(s,s^*)$$   

其中FN代表false negative数量，FP代表false positive数量。从loss的形式来看，分子对所有出现在label的 $s$ 的score及进行了求和，最大化该分数；分母在softmax的原式的基础上增加了一项cost，增大了输出概率分布与FN和FP的margin。    


## 其他方法    

### 引入成分路径特征    
[Encoding Syntactic Constituency Paths for Frame-Semantic Parsing with Graph Convolutional Networks](https://arxiv.org/abs/2011.13210)

#### 短语结构句法树(Constituency tree)      
<div align="center">
<img src=./note_fig/constituency_tree.png width=30% />
</div>

成分句法树表示了句子各个短语成分之间的关系，如上图    
- 自底向上的看（规约），$Rachel$ 组成了名词->名词短语然后表示到句子中， $had$ 作为动词过去式参与到动词短语最终句子由名词短语+动词短语构成。
- 自顶向下（推导），即一个句子从开始符号S，首先推导出`名词短语+动词短语`，接着一步一步往下推导，直到所有符号都是终结符。

#### 图卷积网络(Graph Convolutional Networks)     
<div align="center">
<img src=./note_fig/graphcnn_.png width=60% />
</div>

一般卷积网络中的卷积通过卷积核来进行，而图卷积网络中的卷积依赖于边；CNN中可以认为临近的点都有边相连。对于成分句法树的GCN，可以形式化为： 
    
$H^{(l+1)}=LN(\sigma(AH^lW^l+b^l)) $    

其中，$A$ 是图的邻接矩阵表示，$H^l\in R^{N×D}$ 是第l层各个点状态的表示，$W^l\in R^{D×E}$ 是维度转换的权重矩阵，其中 $H^0=$ n个点的嵌入向量；上式代表的含义就是，该层各点状态计算仅取决于上一层的结果，即对于某个点 $i$ ，其当层的向量表示为所有与 $i$ 相邻点上一层的状态向量求和，再经过线性变化和激活函数。下图是对应与上述成份句法树的GCN计算。    

<div align="center">
<img src=./note_fig/graphcnn.png width=60% />
</div>

#### 获取短语结构特征     
<div align="center">
<img src=./note_fig/constituency_tree.png width=30% />
</div> 

在短语结构句法树中，任意两个叶子结点可以确定唯一最短路径，如图中had->little可以确定经过四个中间结点的路径：`VBD->VP->NP->JJ`，通过GCN最后一层的输出可以提取任意两个词之间的路径信息：   
$p_{i,j}=\sum{c_k} $，其中 $k$ 是 $i$ 与 $j$ 路径上所有中间结点， $c$ 代表GCN最后一层输出   

<div align="center">
<img src=./note_fig/encode_p.png width=50% />
</div>

$a=LN(BiLSTM(e\bigoplus p_{root})+e) $  
$b=LN(BiLSTM(e\bigoplus p_{j})+e) $   
$p_{root}=p_{1,root},...,p_{n,root}$   
$p_{j}=p_{1,j},...,p_{n,j}$，j是句子中能够激发语义框架的`target word`    

上图通过将文本的embedding和短语结构句法树的路径特征结合，得到了两种路径特征表示，以树根为中心的 $a$ 和以`target word`为中心的 $b$ ，以此向基本的文本embedding注入了句法信息
。   
在`Target Identification、Frame Identification`中，使用a作为基本输入特征，而在`Semantic Role Labeling`中不仅使用a，还可以通过b向span的表示中注入每个词到`target word`的短语结构句法的路径信息。


### End to End    
[A Graph-Based Neural Model for End-to-End Frame Semantic Parsing](https://arxiv.org/pdf/2109.12319v1.pdf)

#### Span Representation     
通过`Bert, BiHLSTM`得到输入句子单词级别的特征表示，限制最大长度的前提下枚举句子中所有span，拼接span相关特征向量作为`span representation` $g$。

#### Node Building & Frame Classification of Predicate Nodes      
对每个span进行分类，输入就是每个span对应的 $g_i$ ，经过MLP层对齐到输出类别维度后采用softmax计算各类别的概率。

每个span在一个语义框架下可能的类别如下：
- FPRD: a full predicate span.
- PPRD: a partial predicate span.
- ROLE: a role span.
- NULL: a span that is not a graph node. 

同时一句话中可能包含多个语义框架，所以一个span可能既可以作为激发语义框架的谓词(predicate)、也可能作为语义框架角色(role)或者兼二者于一身，又或者完全不参与任何语义框架。故最终对一个span的分类类别包括8中组合：`{FPRD, PPRD, ROLE, FPRD-PPRD, FPRD-ROLE, PPRD-ROLE, FPRD-PPRD-ROLE, NULL}`。

对分类为predicate node的span，用上述相同的分类方法预测其所激发语义框架，类别数为对应`lexical unit`在Frame Net中可能对应的语义框架种类数。

经过上述处理，可以得到包含若干结点的无边图，其中结点可以分为两大类：`Predicate Nodes`和`Role Nodes`，前者可以认为是该句子中所有语义框架中的`target word`集合，后者是语义框架中`semantic role`集合。

#### Edge Building       
前面其实已经完成了传统方法中的`Target Identification`和`Frame Identification`，并且已经完成了对某个span是否能作为某个语义框架的语义角色的识别，接下来要找到不同结点直接的联系。

一个语义框架由`predicate`和该语义框架下各种`semantic role`组成，其中`predicate`具有主导整个语义框架的作用，故对一个语义框架的识别可以转化为判断上述节点`Predicate Node`和`Role Node`之间是否有关系（边），问题转化为两个结点之间的分类问题，文中构建了下面两种边：

- Predicate-Predicate Edge: 连接两个`PPRD`类别的结点，主要用于处理当一个语义框架的`predicate`不连续的情况，分类结果为`Connected/NULL`。
- Predicate-Role Edge: 对每个`Predicate Nodes`中的结点，枚举`Role Nodes`中的结点，判断两个点是否存在联系，即能否建立一条边，边的类型即为该`role node`在该`predicate node`所激发的语义框架下对应的语义角色。

<div align="center">
<img src=./note_fig/end2end.png width=80% />
</div>

通过对句子中每个span进行特征表示，对每个span分类为`Predicate`和`Role`两类结点，最后找到结点之间的关系（建立边），判断每个`Predicate`结点对应的语义角色；后面图建立的过程均使用通过`Encoding`过程获得的每个span对应的特征向量，从而能够使得这样的端到端模型能够利用到传统三个子任务之间一些相关性的特征。