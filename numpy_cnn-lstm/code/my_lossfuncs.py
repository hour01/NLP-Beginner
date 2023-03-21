import numpy as np

class LossFunc:
    def __init__(self,f,f_derivative):
        self.f = f
        self.f_derivative = f_derivative
    
    def __call__(self,label,predict):
        return self.f(label,predict)
    
    def derivate(self,label,predict):
        return self.f_derivative(label,predict)

# 交叉熵
def crossEntropy(label,predict):
    '''
    predict is excepted the output of softmax (1,classes)
    label is excepted one-hot vector of class (classes,)
    '''
    # batch
    # loss = -np.log(np.max(predict*label,axis=1))
    predict = predict.reshape(-1,1)
    loss = -np.log(predict[np.argmax(label)])
    return loss

def crossEntropy_derivate(label,predict):
    '''
    predict is excepted the output of softmax (classes)
    label is excepted one-hot vector of class (classes)
    '''
    # predict += 0.00001
    label = label.reshape(predict.shape)
    return -label/predict

cross_entropy = LossFunc(
    crossEntropy,
    crossEntropy_derivate
)


def BCEloss(label, predict):
    '''
    predict is excepted the output of softmax (1,classes)
    label is excepted the label 1/0
    '''
    predict = predict.reshape(-1,1)
    loss = -(label*np.log(predict) + (1-label)*np.log(1-predict))
    return loss

def BCEloss_derivate(label, predict):
    label = np.array(label).reshape(predict.shape)
    return label/predict - (1-label)/(1-predict)


# sse
def f_sum_of_squared_error(label,predict):
    return (predict-label)**2

def f_sum_of_squared_error_derivate(label,predict):
    return 2*(predict-label)

# 平方和误差
sse = LossFunc(
    f_sum_of_squared_error,
    f_sum_of_squared_error_derivate
)


if __name__ == '__main__':
    label = np.array([0,0,1])
    predict = np.array([0.3,0.4,0.3])
    print(crossEntropy(label,predict))


