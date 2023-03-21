import numpy as np
## batch

#激活函数
class Func:
    def __init__(self,f,f_derivative,jacobin=False):
        self.f = f 
        self.f_derivative = f_derivative
        self.jacobin = jacobin # True 表明f导数将使用雅克比矩阵进行表示

    def __call__(self,z):
        return self.f(z)

    def derivate(self,z):
        return self.f_derivative(z)

# sigmomid
def f_sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def f_sigmoid_derivative(a):
    y = f_sigmoid(a)
    return y*(1-y)
sigmoid = Func(f_sigmoid,f_sigmoid_derivative)

# relu 
def f_relu(z):
    return np.maximum(z, 0)

def f_relu_derivate(a):
    return np.heaviside(a,0.5)

relu = Func(f_relu,f_relu_derivate)

# softmax 
def f_softmax(z):
    '''
    the input excepted (n_hidden)
    '''
    # 直接使用np.exp(z)可能产生非常大的数以至出现nan
    # 所以分子分母同时乘以一个数来限制它
    # 这里用 exp(-np.max(z))

    #batch
    # exps = np.exp(z-np.max(z))
    # exp_sum = np.sum(exps,axis=1,keepdims=True)
    # print(z)
    exps = np.exp(z-np.max(z))
    exp_sum = np.sum(exps)
    # print(exps/exp_sum)
    return exps/exp_sum


# 得到一个对称矩阵，[i,j]代表第j个输出对应第i个输入的偏导数值
def f_softmax_derivate(z):
    '''
    the input excepted (n_hidden)
    output: (n_hidden, n_hidden)
    '''
    #batch
    # res = []
    # y = f_softmax(z)
    # for line in y:
    #     res.append(np.diag(line)-line.reshape((-1,1)).dot(line.reshape(1,-1)))
    # return np.array(res)
    y = f_softmax(z).reshape((-1,))
    return np.diag(y)-y.reshape((-1,1)).dot(y.reshape(1,-1))
# softmax 导数只能用雅克比矩阵表示，无法简化
softmax = Func(f_softmax,f_softmax_derivate,True) 


if __name__=="__main__":
    # z = np.array([[0.3,0.4,0.3],[0.3,-0.4,0.3]])
    # print(sigmoid(z),f_sigmoid_derivative(z))
    # print(f_relu(z),f_relu_derivate(z))
    z = np.array([1,1])
    # z = f_softmax(z)
    # print(z)
    print(f_softmax_derivate(z))
    exit()
    ##test bp
    da = np.array([[1,0,1],[0,1,0]])
    da_dz = f_softmax_derivate(z)
    dc_dz = []
    # batch
    for i in range(0,len(da)):
        dc_dz.append(np.sum(da[i]*da_dz[i],axis=1))
    dc_dz = np.array(dc_dz)
    print(dc_dz)
    print(np.sum(da[1]*da_dz[1],axis=1))
