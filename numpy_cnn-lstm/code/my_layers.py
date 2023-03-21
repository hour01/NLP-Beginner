import numpy as np
from abc import abstractmethod
from my_funcs import Func


class Layer:
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, d: np.ndarray) -> np.ndarray:
        pass

# 用于reshape时的反向传播
class ReshapeLayer(Layer):
    def __init__(self, from_shape, to_shape):
        self.from_shape = from_shape
        self.to_shape = to_shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.from_shape is None:
            self.from_shape = x.shape
        return x.reshape(self.to_shape)

    def backward(self, d: np.ndarray) -> np.ndarray:
        return d.reshape(self.from_shape)


class FuncLayer(Layer):
    def __init__(self, activate_fn: Func):
        self.f = activate_fn
        self.z: np.ndarray = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.z = x
        return self.f(x)

    def backward(self, dc_da: np.ndarray) -> np.ndarray:
        da_dz = self.f.derivate(self.z)
        if self.f.jacobin:
            # 如果求导结果只能表示成雅克比矩阵，得使用矩阵乘法
            #dc_dz = dc_da.dot(da_dz.T)
            dc_dz = []
            # batch
            # 见笔记
            # for i in len(dc_da):
            #     dc_dz.append(np.sum(dc_da[i]*da_dz[i],axis=1))
            # dc_dz = np.array(dc_dz)
            dc_dz = dc_da.dot(da_dz.T)
        else:
            # 求导结果为对角矩阵，可以采用哈达马积（逐值相乘）来简化运算
            dc_dz = dc_da * da_dz
        return dc_dz


# 全连接层
class FullConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # 把输入当作 1*input_size 的 行向量
        # 输出为 1*output_size
        self.w = np.random.normal(loc=0.0, scale=1.0, size=(self.input_size, self.output_size))
        self.b = np.random.normal(loc=0.0, scale=1.0, size=(1, self.output_size))
        self.x: np.ndarray = None  # input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(1, -1)
        self.x = x
        self.z = x.dot(self.w) + self.b
        return self.z

    # dc_dz 形状和输出一致，
    def backward(self, dc_dz: np.ndarray) -> np.ndarray:
        # 反向传播
        dc_dx = dc_dz.dot(self.w.T)
        # 计算当前W权重变化
        self.w += self.x.T.dot(dc_dz)
        self.b += dc_dz
        return dc_dx


class MaxPoolingLayer(Layer):
    '''
    A simple maxpooling with no stride, map all hidden unit into one 
    '''
    def __init__(self):
        super().__init__()
        self.argmax = None
        self.inputshape = None

    def __call__(self,z:np.ndarray):
        '''
        the input is expected (kernels, len) 
        '''
        self.argmax = np.argmax(z,axis=1)
        self.inputshape = z.shape
        return np.max(z,axis=1)
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        '''
        the input is expected in 1-D
        '''
        d_input = np.zeros(self.inputshape)
        d_input[range(0,len(d_input)),self.argmax] = d
        return d_input 

class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    def __call__(self, X, training=True):
        c = (1 - self.p)
        if training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward(self, d):
        return d * self._mask

class TextCNNLayer(Layer):
    '''
    a simple textcnn with 1 input channels, 1 stride, (x,embeding_size) ketnel size
    '''
    def __init__(self, output_channels, kernel_size):
        self.output_channels = output_channels
        self.kernel = np.random.normal(loc=0.0, scale=1.0, size=(
           output_channels, kernel_size[0], kernel_size[1]))
        self.b = np.random.normal(loc=0.0, scale=1.0, size=(self.output_channels, 1))
        self.input = None
        self.padding_num = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        the input is expected as (len, hidden)
        '''
        assert x.shape[1] == self.kernel.shape[2]
        # assert x.shape[0] >= self.kernel.shape[1]

        self.input = x

        # padding
        if self.input.shape[0] < self.kernel.shape[1]:
            self.padding()
        
        res = np.zeros((self.kernel.shape[0],self.input.shape[0]-self.kernel.shape[1]+1))
        for i in range(0,res.shape[0]):
            for j in range(0,res.shape[1]):
                res[i,j] = np.sum(self.kernel[i]*self.input[j:j+self.kernel.shape[1],:])
        res += self.b
        return res
    
    def backward(self, d: np.ndarray) -> np.ndarray:
        '''
        the input is expected as the same shape of Text-CNN's output (output_channels, len-k+1)
        '''
        res = np.zeros(self.input.shape)
        dw = np.zeros(self.kernel.shape)
        # each channel
        for i in range(0,len(self.kernel)):
            for j in range(0,len(d[i])):
                res[j:j+self.kernel.shape[1]] += d[i,j]*self.kernel[i]
                dw[i] += d[i,j]*self.input[j:j+self.kernel.shape[1]]
        # print('grad of kernel')
        # print(dw)
        self.kernel += dw
        self.b += np.sum(d,axis=1,keepdims=True)
        # print('grad of bias')
        # print(np.sum(d,axis=1,keepdims=True))
        if self.padding_num != 0:
            return np.delete(res,list(range(self.input.shape[0]-self.padding_num,self.input.shape[0]-1)),axis=0)
        return res
    
    def padding(self):
        self.padding_num = self.kernel.shape[1] - self.input.shape[0]
        self.input = np.row_stack((self.input, np.zeros((self.padding_num,self.input.shape[1]))))


if __name__ == '__main__':

    # for maxpooling
    # print('maxpooling')
    # max = MaxPoolingLayer()
    # m = np.array([[2,3,6],[3,3,4],[8,1,1]])
    # print(max(m))
    # print(max.backward(np.array([1,1,1])))

    # for conv
    print('cnn')
    cnn = TextCNNLayer(2,(2,4))
    print('initializing kernel')
    print(cnn.kernel)
    print('initializing bias')
    print(cnn.b)
    input = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    print('output of cnn_forward')
    print(cnn(input))
    print('sum of kernel for each channel')
    print(np.sum(cnn.kernel[0]),np.sum(cnn.kernel[1]))

    # for conv bp
    da = np.array([[1,1],[1,1]])
    print('bp of cnn')
    print(cnn.backward(da))
