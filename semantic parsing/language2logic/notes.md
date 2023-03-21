# Semantic Parsing - Language to Logic Form

## 任务目标      
将自然语言句子转换成计算机可以理解的形式化语义表示，例如SQL语句、lambda表达式等。本次实验实现了将自然语言句子转化为lambda表达式。    
参考[Language to Logical Form with Neural Attention](https://arxiv.org/pdf/1601.01280.pdf)，使用pytorch实现了基本的seq2seq模型，以及seq2seq with attention模型。

>下面是一个自然语言转成lambda表达式的例子：       
`约翰出生在哪个城市？`     
lambda表达式：$\lambda x City(x)\land BirthPlace(John, x)$  

## 数据集介绍
ATIS数据集：包含了对航班预订系统的5410个查询，每个句子与其对应的lambda表达式组成一对。     
实际实验中直接下载使用已经预处理过的数据。[Language to Logical Form with Neural Attention](https://arxiv.org/pdf/1601.01280.pdf)      
其中地名、航班、日期等实体信息都被替换为抽象符号，例如下面例句，`ci0,ci1`就是对地名进行抽象。    
>show me all flight from ci0 to ci1    
lambda表达式：( lambda \$0 e ( and ( flight \$0 ) ( from \$0 ci0 ) ( to \$0 ci1 ) ) )

## Method     
将该语义解析任务当作机器翻译任务进行处理，将逻辑表达式看作一系列token，使用encoder-decoder结构，将问题转化为sequence to sequence问题。

### Naive Sequence to Sequence (NS2S)

<div align="center">
<img src=./note_figures/seq2seq.png width=40% />
</div>

- 使用两层的LSTM作为encoder和decoder；encoder编码输入序列，将最后一个时刻LSTM的(h_n,c_n)作为decoder初始状态(h_0,c_0)
- 在序列加入\<s>和\</s>作为序列开始和结束的符号
- 训练阶段，解码时采用`Teacher forcing`策略，即输入给decoder的序列是真实标注序列；在预测阶段再采用自回归的生成方式。

### Sequence to Sequence with attention(S2SA)      
与上述`NS2S`的区别在于`NS2S`直接采用decoder的输出进行token预测，而`S2SA`将decoder的输出与encoder的输出做了一次attention后再进行预测；从而能够在预测某个token时，增加encoder输出与其相关的上下文信息（词语到词语表示之间的软对齐），从而更好的做出预测。

- attention：对于每个decoder的输出 $o_i$ ，计算所有encoder输出 $s$ 与其的attention分数，对所有encoder输出依据该分数加权求和，再与该decoder输出拼接作为最终预测向量。

<div align="center">
<img src=./note_figures/attention.png width=60% />
</div>

## 结果     
计算模型预测结果的正确率，其中只有一个句子对应的logic form所有token均预测正确才计数。    
|method|ACC(%)|
|-|-|
|NS2S|61|
|S2SA|82|

## 其他方法

### seq2seq的问题     
- 语义解析的目标语言大多具有层次结构，但seq2seq无法对此进行建模，只能把目标语言当作扁平序列 
- seq2seq解码过程需要长距离依赖 

<div align="center"> 
<img src=./note_figures/hierarchical.png width=60% />
</div>

如上图所示，需要筛选的航班信息条件为 `出发时间>4 pm` and `from dallas to san francisco`；结构化表示中    
- 第一层确定了lambda表达式基本结构，筛选符合\<n>中条件的航班
- 第二层确定限制条件的个数
- 后续再通过\<n>逐步补充所有限制条件

### Seq2Tree
[Language to Logical Form with Neural Attention](https://arxiv.org/pdf/1601.01280.pdf) 
<div align="center">
<img src=./note_figures/seq2tree.png width=60% />
</div>

层次化decoder，\<n>表示非终结符，生成层次化结构表示    
- 非终结符存放到队列中，一直解码直到队列为空

### Coarse-to-fine      
[Coarse-to-Fine Decoding for Neural Semantic Parsing](https://arxiv.org/pdf/1805.04793.pdf)

- 自然语言和目标logic form差距太大，先生成sketch，再在sketch的帮助下，与文本encoder输出一起完成对最终logic form的生成。
- 生成sketch的过程相比直接生成完整语义要简单，而且很多句子的语义结构相似（根据时间、出发地、目的地检索信息等），具有类似的sketch

<div align="center">
<img src=./note_figures/coarse2fine.png width=80% />
</div>

最终的decoder既依赖于第一层文本encoder的输出，又依赖于sketch的encoder输出，即  

$$h_t=LSTM(h_{t-1},i_t) $$

$$
i_t=\left\{
\begin{matrix}
 v_k , y_{t-1} \quad is\quad determined\quad  by\quad  a_k \\
 y_{t-1} , \quad  otherwise
\end{matrix}
\right.
$$
关键在于 $i_{t-1}$ 的选择，若 $y_{t-1}$ 出现在sketch中，则decoder下一状态的输入采用sketch的encoder的输出，否则采用 $y_{t-1}$ ，就是对自回归生成的改进。