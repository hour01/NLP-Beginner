# NLP-Beginner 

NLP入门基本训练，主要包括相关深度学习基础知识的学习、NLP中常用基础模型代码实践。

- `ELMo、Bert、Transformer`文件夹下主要是相关模型的学习理解
- `numpy_CNN-LSTM`文件夹下是使用numpy分别实现了`Text-CNN`和`LSTM`，可以实现正向传播和反向传播更新参数。
- `word2vec`文件夹下主要是对word2vec的学习，实现了`Skip Gram`模型进行小实验
- `Text Classification`主要是实现了`最大熵、TextCNN、LSTM、Bert`等模型用于文本分类
- `Named Entity Recognition`下主要实现了NER任务的基础模型，包括`BiLSTM-CRF、BERT-BiLSTM-CRF、Char_emb-BiLSTM-CRF`。
- `Chinese Words Segmentation`主要实现了将`BiLSTM-CRF、Bert-BiLSTM-CRF`用于中文分词进行实验。
- `semantic parsing`主要包含两个子任务的实现，一个是`sentence to logic form`，另一个是`frame semantic parsing`，分别对两篇文章的模型进行复现。