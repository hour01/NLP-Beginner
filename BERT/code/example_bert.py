from transformers import BertTokenizer, BertModel, BertTokenizerFast
import torch

'''
Pretrained model on English language using a masked language modeling (MLM) objective.
It was introduced in this paper and first released in this repository.
This model is uncased: it does not make a difference between english and English.
'''

# 传入的参数是包含模型所有文件的目录名。
# 其中vocab文件的文件名必须是vocab.txt文件名，模型文件名必须是pytorch_model.bin，配置文件名必须是config.json.
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
model = BertModel.from_pretrained("./bert-base-uncased")


# encode仅返回input_ids
# encode_plus返回所有编码信息
# ‘input_ids’：是单词在词典中的编码
# ‘token_type_ids’：区分两个句子的编码（上句全为0，下句全为1）
# ‘attention_mask’：指定对哪些词进行self-Attention操作
# 整个输入最开始会被加上<CLS>, 每句话最后会被加上<SEP>
# 都会被转小写
print(tokenizer.encode('hello world !'))                    #[101, 7592, 2088, 999, 102]
print(tokenizer.encode('mid-twenties'))                    #[101, 3054, 1011, 18946, 102]
# bert 会把其拆分为 mid - twentied 三个词
sen_code = tokenizer.encode_plus('there is a dog','it is a cat')
print(sen_code)
# dict type
# {'input_ids': [101, 2045, 2003, 1037, 3899, 102, 2009, 2003, 1037, 4937, 102], 
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# 将input_ids转回token
print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))
# ['[CLS]', 'there', 'is', 'a', 'dog', '[SEP]', 'it', 'is', 'a', 'cat', '[SEP]']

# 对编码进行转换，以便输入Tensor
tokens_tensor = torch.tensor([sen_code['input_ids']])       # 添加batch维度并,转换为tensor,torch.Size([1, 11])
segments_tensors = torch.tensor(sen_code['token_type_ids']) # torch.Size([11])
 
model.eval()
 
# 进行编码
with torch.no_grad():
    
    outputs = model(tokens_tensor, token_type_ids = segments_tensors)
    encoded_layers = outputs   # outputs类型为tuple
    #(last_hidden_state, pooled_output, (hidden_states), (attentions))
    
    print(encoded_layers[0].shape, encoded_layers[1].shape) 
          #encoded_layers[2][0].shape, encoded_layers[3][0].shape)
    # torch.Size([1, 11, 768]) torch.Size([1, 768])
    # torch.Size([1, 11, 768]) torch.Size([1, 12, 11, 11])

# 以输入序列为11为例：

# last_hidden_state.Size([1, 11, 768])
# 输出序列

# pooler_output：torch.Size([1, 768])
# <CLS>最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的（BertPooler）

# (hidden_states)：tuple, 13 * torch.Size([1, 11, 768]) 
# 隐藏层状态（包括Embedding层），取决于 model_config 中的 output_hidden_states
# 它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，
# 每个元素的形状是(batch_size, sequence_length, hidden_size)

# (attentions)：tuple, 12 * torch.Size([1, 11, 11])
# 注意力层，取决于 model_config 中的 output_attentions
# 是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值。

# 对于单句输入的batch
# use batch_to_ids to convert sentences to character ids
sentence_lists = ['I am eatting an apple .', 'history of apple computers .'] #references
tmp = tokenizer(sentence_lists, padding=True, return_tensors="pt")
print(tmp)
# {'input_ids': tensor([[ 101, 1045, 2572, 4521, 3436, 2019, 6207, 1012,  102],
#                       [ 101, 2381, 1997, 6207, 7588, 1012,  102,    0,    0]]), 
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
#                           [1, 1, 1, 1, 1, 1, 1, 0, 0]])}
outputs = model(tmp['input_ids'])
print(outputs[0].shape, outputs[1].shape)
#torch.Size([2, 9, 768]) torch.Size([2, 768])


tokenizer=BertTokenizerFast.from_pretrained('./bert-base-uncased')
sentence_lists = 'AL-AIN , United Arab Emirates 1996-12-06'#'I am eatting an apple .' #references
tmp = tokenizer.encode_plus(sentence_lists, padding=True, return_tensors="pt",return_offsets_mapping=True)
print(tmp)
print(tokenizer.convert_ids_to_tokens(tmp['input_ids'][0]))
# {'input_ids': tensor([[ 101, 1045, 2572, 4521, 3436, 2019, 6207, 1012,  102]]), 
# 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]), 
# 'offset_mapping': 
# tensor([[[ 0,  0],
#          [ 0,  1],
#          [ 2,  4],
#          [ 5,  8],
#          [ 8, 12],
#          [13, 15],
#          [16, 21],
#          [22, 23],
#          [ 0,  0]]])}
# 其中每一个[i,j] 代表对应的ids为原输入字符串中的索引位置text[i,j]
# ['[CLS]', 'i', 'am', 'eat', '##ting', 'an', 'apple', '.', '[SEP]']
# 对于批处理可以使用tokenizer.batch_encode_plus