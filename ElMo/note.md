# ELMo    
  
>word2vec、Glove得到的词向量表示有一个很大的局限性，一个单词被表示为一个固定的向量，这个向量没有在不同语境中的灵活性，即无法解决多义词的问题，一个经典的例子就是apple根据语境的不同，可以理解为一种水果或者公司

## ELMo的网络结构   
<div align="center">
<img src=./note_fig/ELMo.png width=70% />
</div>

上图是在计算得到token上下文表示词向量的计算图，其中每个虚线框代表一层LSTM   
<div align="center">
<img src=./note_fig/ELMo_train.png width=70% />
</div>

上图是使用ELMo在预训练时使用到的结构    
其中 Char Encoder Layer使用的是下文中的 cnn-big-lstm    

[Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf)    
即在字符级上用类似Text-CNN的方法做embedding     
最后输出的词向量是维度是每层双向LSTM的连接产生的维度，把相同word embedding做个concat，与双向lstm输出的维度统一起来。    

## ELMo的双向语言模型   

ELMo基于双向LSTM的语言模型，即biLM

**前向的LSTM语言模型**可以表示为如下形式：      
   
$P(w_1,w_2,w_3,...,w_N)=\prod_{k=1}^N p(w_k|w_1,w_2,...,w_{k-1}) $ 
   
**后向的LSTM语言模型**可以表示为如下形式：   
  
$P(w_1,w_2,w_3,...,w_N)=\prod_{k=1}^N p(w_k|w_{k+1},w_{k+2},...,w_N) $

**ELMo的训练目的是最大化前向和后向的似然概率:**       
      
$$\sum_{k=1}^N(logP(w_k|w_1,...,w_{k-1};\Theta_x,\overrightarrow{\Theta}_{LSTM},\Theta_s) + logP(w_k|w_{k+1},...,w_N;\Theta_x,\overleftarrow{\Theta}_{LSTM},\Theta_s))  $$        

$其中，\overrightarrow{\Theta}_{LSTM}表示前向LSTM的网络参数，\overleftarrow{\Theta}_{LSTM}表示后向LSTM的网络参数，\Theta_x是token表示层的参数，\Theta_s为bi-LSTM后的线性变化的参数，后面两者是各层bi-LSTM共享的$  

## ELMo表示上下文词向量   

对于每个token，经过L层双向LSTM语言模型后，一共有 $2*L+1$ 个表征：      
$$R_k=\{x_k,\overrightarrow{h}_{k,j},\overleftarrow{h}_{k,j}|j=1,...,L\}$$   
其中，k表示第k个token(第K个时间)，x表示word_emb，h表示没层LSTM的输出     
也可以将每层LSTM的的输出表示为 
$$h_{k,j}=[\overrightarrow{h}_{k,j}:\overleftarrow{h}_{k,j}]$$   

如何从这么多的表示中得到token的上下文向量？最简单的方法就是选择顶层LSTM的输出。同时也有另一种方法表示为：  
$ELMo_k^{task}=E(R_k;\Theta^{task})=\gamma^{task}\sum_{j=0}^Ls_j^{task}h_{k,j}  $    
对于每层向量，我们加一个权重 $s_j$ （一个实数），将每层的向量与权重相乘，然后再乘以一个权重 $\gamma^{task}$ 。每层 LSTM 输出，或者每层 LSTM 学到的东西是不一样的，针对每个任务每层的向量重要性也不一样，所以有L层 LSTM，L+1个权重，加上前面的 $\gamma^{task}$，一共有L+2个权重。 权重 $\gamma^{task}$ 是对于训练多个任务时加入的，对于一个单独的任务，就不需要这个参数了。

## ELMo的效果  

#### 在语义消歧(word sense disambiguation)上   

使用越高层的表示效果越好，说明越高层，对语义理解越好    

#### 在词性标注(POS tagging)   

第一层好于第二层，表明低层更能表示句法信息，词性信息。


## 使用   

见`./code/example.py`
官方首页: https://allennlp.org/elmo
源码：https://github.com/allenai/allennlp 
多语言：https://github.com/HIT-SCIR/ELMoForManyLangs 