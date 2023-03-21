'''
a simple example of using elmo word vector
'''
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "options.json" # 配置文件
weight_file = "weights.hdf5" # 权重文件

# 这里的1表示产生一组线性加权的词向量。
# 如果改成2 即产生两组不同的线性加权的词向量。
elmo = Elmo(options_file, weight_file, 1, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentence_lists = [['I', 'am', 'eatting', 'an', 'apple', '.'], ['history', 'of', 'apple', 'computers', "."]] #references

character_ids = batch_to_ids(sentence_lists)

embeddings = elmo(character_ids)['elmo_representations'][0]

print(embeddings[0][4],embeddings[1][2])
print(embeddings.size())
print(type(embeddings))
# 输出
# tensor([ 0.5961, -0.4167,  0.8023,  ...,  0.1543,  0.9495,  0.5093],
#        grad_fn=<SelectBackward0>) 
# tensor([ 0.6066, -0.3123,  0.7796,  ...,  0.4197, -0.0687,  0.3776],
#        grad_fn=<SelectBackward0>)
# torch.Size([2, 6, 1024])
# <class 'torch.Tensor'>