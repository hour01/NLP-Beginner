# 最大熵文本分类

# 参考文献
[Using Maximum Entropy for Text Classification](http://www.kamalnigam.com/papers/maxent-ijcaiws99.pdf)

# 实验
## 数据预处理，提取n-gram
`./code/preprocess.cpp`主要对数据进行预处理，抽取出`sentence-sentiment`语句情感类别数据对，同时枚举出n-gram词库（字典中最大连续单词数为n），输出到`./code/processed_data_n-gram`。

## 最大熵模型的实现 
`./code/max_entropy_textclassification_numpy.py`
因为安装`c++ numpy`时出现一些问题一下没解决，且感觉对于该问题，使用python/c++在实现难度上并没有很大区别，故放弃使用c++完成最大熵模型。
