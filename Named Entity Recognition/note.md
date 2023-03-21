# NER
## 数据集
[conll03](https://www.clips.uantwerpen.be/conll2003/ner/)
|train|validation|test|
|----|----|----|
|14041|3250|3453|
其中在ner任务中只使用其第一列(单词)和第四列(ner tag) 
原数据集采用的是BIO1标注方法 ,BIO2标注法则要求任何entity都必须以B开头，包括单个词的entity 
> The chunk tags and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. 
testb line 37111
> The DT I-NP O
Thai NNP I-NP I-MISC
Commerce NNP I-NP I-ORG
Ministry NNP I-NP I-ORG 
 
- persons:`'B-PER','I-PER'`
- organization: `'B-ORG', 'I-ORG'`
- location: `'B-LOC', 'I-LOC'`
- miscellaneous entities: `'B-MISC', 'I-MISC'` 

即一个句子中只会出现以下几种情况的phrase：
`(I-type),(I-type,...,I-type),(B-type,I-type,..,I-type),(B-type)`
其中后两种情况需要两个相同类别的entity紧紧相邻 
 
使用以下数字代表相应tag 
{'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

#### Metric:
$precision=correct_{num}/predict_{num}$
$recall=correct_{num}/gold_{num}$
$f1=(2*precision*recall)/(precision+recall)$

$gold_{num}$: 数据集标签中出现的entity 
$predict_{num}$: 模型一共产生的预测总数
$correct_{num}$: 预测准确的数量 
上述entity数量均不包括`'O'`，且所得到的phrase一定是合法的 
例如：I-PER I-ORG这样的序列不合法，I-ORG不会被计入， O O B-PER同样不合法 

官方提供的conlleval.pl测试程序(使用的标注方法为IOB2): 输入文件至少提供真实tag和预测tag，其中每一行预测tag为最后一列。 
但实验使用的数据集是IOB1标注法，故使用自己实现的`utils.py/Metric` 

### 词向量选择 
对于NER任务，猜测大小写会对性能产生影响，故采用多种预处理进行初步实验 
- 区分大小写，使用Google News预训练词向量，其中OOV达到：8372/30291 
- 全部转小写，使用glove预训练词向量，其中OOV达到：3924/26871 

## BiLSTM 
将BiLSTM的实验结果作为本次实验的baseline。

具体实现：采用BiLSTM的每时刻的对应输出加上一层线性变换得到每个词对应各种tag的概率，最后使用交叉熵损失函数计算loss进行训练 


**随机初始化词向量** 
|大小写方式|val_F1|test_F1|
|-|-|-|
|统一小写|65|59.41|
|区分大小写|75|64.75|

**使用Google News预训练词向量** 
|大小写方式|val_F1|test_F1|
|-|-|-|
|统一小写|78|71.93|
|区分大小写|88|80.52|

经过验证，是否区分大小写在conll03数据集上对于NER任务有影响。 

## BiLSTM-CRF 

使用Google News预训练词向量，且区分大小写 
简单的在BiLSTM的基础上增加CRF层 

|词向量初始化|val_F1|test_F1|
|-|-|-|
|随机初始化|78|68.06|
|Google News|88|80.67|

- 对于不使用预训练词向量的情况，加上CRF层对结果有不小提升
- 但对于使用预训练词向量初始化的情况，模型获得的提升很小 

通过查看在测试集上预测错误的案例，发现绝大部分错误是：
- 模型根本没有将某个词识别为entity 
- 模型错误的将未标识的词识别为某个类别的entity 

猜测BiLSTM解决NER任务时，模型缺陷更多在于对词语所处上下文的语义理解，并非是CRF所希望的赋予模型输出tag序列时的容错率 
同时对于在conll03上的NER任务，其类别数较少，且连续出现entity的占比较少，故CRF也许无法得到完全训练  

>Glove 300d
val_F1≈81, test_F1=72.31

>Glove 100d
val_F1≈84, test_F1=76.65 


## char_emb-BiLSTM-CRF 
输入LSTM的词向量中，除了包括预训练词向量，还包括了字符级的词向量，将二者拼接起来输入LSTM，通过对单词进行字符级embedding，可以一定程度上增强模型泛化能力，即在测试时遇到\<UNK> 时也能有效提取特征。 
- **Char-CNN**：建立字符级词表，先对字符进行embedding，随后在此基础上使用kernel_size为`(3,embedding)`的kernel进行卷积，最后对每个output_channel做max_pooling连接。
- **SGD**：使用`torch.optim.SGD`进行优化，其性能表现优于以往默认使用的`Adam`；同时使用weight decay:$\alpha_t=\alpha_0/(1+0.05t)$，及梯度裁剪(gradient clipping)，max_norm = 5.0。

>使用Glove 100d 初始化，char_emb=25，hidden=125 
val_F1≈88，test_F1=83.04

>使用Glove 100d 初始化，char_emb=25，hidden=100 
val_F1≈88，test_F1=81.04 

>Glove 100d with SGD lr=0.015，hidden = 200
val_F1≈92, test_F1=87.21 



## Bert-CRF 
使用hugging face 提供的`bert-base-uncased`初始化bert
遇到的问题：因为bert采用字符级encode，会将输入token进行拆分，故经过tokenizer后token数量会增加，在conll03中包括：
- 将`'mid-twenties'`拆为`'mid', '-', 'twenties'` 
- 将`javasoft`拆为`'java', '##so', '##ft'` 
即会增加输入句子token数量，造成与实际ner_tag数量不对等，一般有以下几种解决方法： 
①按照token预测结果进行打分； 
②只要词语中有一个token被预测为entity的一部分，其他所有成分都当作entity的一部分； 
③训练和decode时都以每个单词第一个token为准 

解决方法如下： 
- 根据bert-encoder输出结果，将每个子词都给予对应父词的标签用于训练 
- 同时记录每个样本的offest_mapping，即被拆分词的位置 
- 计算F1时将序列整合到最初的长度，整合过程中对每个被分开的词，选取其子词中频数最高的预测标签作为其预测标签(投票法) 

>使用bert-base-uncased（不进行微调） 
lr=0.01,batch=16  
val_F1≈84，test_F1=82


## Bert-BiLSTM-CRF 
即在Bert后加上BiLSMT-CRF, 把Bert当作embedding层 

>使用bert-base-uncased （不进行微调） 
lr=0.01,batch=16,hidden=150 
val_F1≈90，test_F1=86.69 


## 结果面板
||%P|%R|%F1|
|-|-|-|-|
|BiLSTM|80.19|80.58|80.52|
|BiLSTM-CRF|80.31|81.03|80.67|
|char_emb-BiLSTM-CRF|87.64|86.77|87.21|
|Bert-BiLSTM-CRF|87.17|86.23|86.69|
