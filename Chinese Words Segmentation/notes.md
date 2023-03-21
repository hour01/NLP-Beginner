# 中文分词

## 任务介绍   

即在中文语句中，将具有独立语义的词语分开，即在词语之间加上边界。而具体可以将其看作NER任务，为每一个token打上标签：
- B: Begin of words，即词的起始字。
- M: Middle of words，即词的非起始字和结束字的其它字。
- E: End of words，即词的结束字。
- S: Single words，即单字。
> 例子：
全 世 界 的 人 民    
B  M  E  S  B  E   


## 数据集   

本实验数据集来源于[SIGHAN 2005第二届中文分词任务（SIGHAN-2nd International Chinese Word Segmentation Bakeoff）](https://aclanthology.org/I05-3017.pdf)中的 Peking University 数据集（简称PKU数据集）。     
数据集分为训练集和测试集，其中每行一句话，不同词组通过两个空格分开。

## Method   

只需要对数据集进行上述BMES标注后即可得到`语句-标注`序列对，采用NER中的方法进行监督训练。

使用了两种模型进行实验
- BiLSTM-CRF  
- BERT-BiLSTM-CRF

其中因为BERT限制最大句子长度为512，故对长句进行切分；同时也测试了对Bert进行微调和不进行微调的两种情况。

## Metric   

在词语级别上进行统计，计算分词结果的准确率，召回率和F1 score。
1. 先将id形式的gold标签序列和pred标签序列转化成符号标签
2. 将两个符号标签序列转化成chunk序列（简单判断B前缀即可）
3. 根据gold chunk sequence生成集合gold_set，遍历pred chunk sequence，如果当前chunk存在于gold_set，correct加1
4. 准确率等于correct除以pred chunk sequence的长度，召回率等于correct除以gold chunk sequence的长度

## 结果

|Model|F1|P|R|
|-|-|-|-|
|BiLSTM-CRF|0.923|0.933|0.913|
|Bert-BiLSTM-CRF|0.948|0.954|0.942|
|Bert-BiLSTM-CRF-Fine_tune|0.965|0.969|0.961|

## 其他方法  

### Wordhood Memory Networks  

[Improving Chinese Word Segmentation with Wordhood Memory Networks](https://aclanthology.org/2020.acl-main.734/)

<div align="center">
<img src=./note_figures/wordhood_memory.png width=70% />
</div>

核心思想就是在传统的encoder-decoder中夹上一层`Memory Networks`，提取wordhodd特征，对于Wordhood特征提取步骤如下。  

#### Lexicon Construction  
 
构建图中 $\mathcal{N}$ ，即提取句子中所有可能的`n-gram`词表。  

#### Wordhood Memory Networks 

- **N-gram Addressing**：   
  对于句子中每一个汉字 $x_i$ ，从 $\mathcal{N}$ 中得到包含 $x_i$ 的所有n-gram短语。   
  例如对图中“民”可以得到$ K=[“民”，“居民”，“民生”，“居民生活”] $    
  将这些n-gram输入embedding后再与encoder传来的 $h_i$ 做softmax得到一个概率分布，即**衡量了一个字和这些短语的相关程度**：    
  $p_{i,j}=\frac{exp(h_i·e_{i,j}^k)}{\sum{exp(h_i·e_{i,j}^k)}} $   
  此处的 $e_{i,j}^k$ 代表上述 $K$ 中n-gram的嵌入向量   
- **Value Reading**：      
  每个字在不同的n-gram中的位置不同，所以需要映射的值也不同，这里使用B I E S标记法：(B:begin ，I:inside，E:end，S:single)，对于上面的例子，得到的value集合为：$K=[V_S，V_E，V_B，V_I]$，将每个value进行embedding后得到 $e_{i,j}^v$ ，使用`N-gram Addressing`得到的概率进行加权求和输出融合了wordhood信息后每个字的特征：
  $o_i=\sum p_{i,j}e_{i,j}^v$
  $a_i=W·(h_i+o_i)$

类似于Attention，对于某个字 $x_i$ ，将encoder的输出 $h_i$ 作为`query`，包含该字的n-gram短语集的embedding作为`key`，该字对应该短语集中每个短语的V作为`value`。   
#### 分析     

<div align="center">
<img src=./note_figures/heatmap.png width=70% />
</div>

上述例子`“他从小学电脑技术”`，在分词任务中把`“从小”`和`“小学”`区分正确是一个困难点     
从上述热力图分析可以看出，在相关性计算中，模型给`“从小”`一个较高的权重，促使模型分类正确。   

直观来看，该方法有效的原因可能在于其能够对句子中哪些`n-gram`的组成对句子语义有更好的表示进行建模，即利用了`wordhood`信息（成词性）。    


