import numpy as np

def softmax(x):
    '''
    the input excepted as (classes, batch)
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 得到一个对称矩阵，[i,j]代表第j个输出对应第i个输入的偏导数值
def softmax_derivate(z):
    '''
    the input excepted as (classes, batch)
    output: (classes, classes, batch)
    '''
    res = np.zeros((z.shape[0],z.shape[0],z.shape[1]))
    y = softmax(z)
    for i in range(0,z.shape[1]):
        res[:,:,i] = np.diag(y[:,i])-y[:,i].reshape((-1,1)).dot(y[:,i].reshape(1,-1))
    return res


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 交叉熵
def crossEntropy(label,predict):
    '''
    predict is excepted the output of softmax (classes, batch)
    label is excepted one-hot vector of class (classes, batch)
    '''
    loss = np.sum(-np.log(np.max(predict*label,axis=0)))/predict.shape[1]
    return loss

def crossEntropy_derivate(label,predict):
    '''
    predict is excepted the output of softmax (classes, batch)
    label is excepted one-hot vector of class (classes, batch)
    '''
    return -label/predict

class LSTM:

    def __init__(self, hidden_dim, input_dim, batch_size, classes, lr = 0.001) -> None:
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.batch = batch_size
        self.classes = classes
        self.alpha = lr

        np.random.seed(1)
        self.Wf = np.random.normal(loc=0.0, scale=1.0, size=(self.hidden_dim, self.input_dim+self.hidden_dim))
        self.bf = np.zeros((self.hidden_dim, 1))
        self.Wi = np.random.normal(loc=0.0, scale=1.0, size=(self.hidden_dim, self.input_dim+self.hidden_dim))
        self.bi = np.zeros((self.hidden_dim, 1))
        self.Wc = np.random.normal(loc=0.0, scale=1.0, size=(self.hidden_dim, self.input_dim+self.hidden_dim))
        self.bc = np.zeros((self.hidden_dim, 1))
        self.Wo = np.random.normal(loc=0.0, scale=1.0, size=(self.hidden_dim, self.input_dim+self.hidden_dim))
        self.bo = np.zeros((self.hidden_dim, 1))

        # 
        self.Wy = np.random.normal(loc=0.0, scale=1.0, size=(self.classes, self.hidden_dim))
        self.by = np.zeros((self.classes, 1))
    
    def lstm_cell_forward(self, xt, a_prev, c_prev):
        '''
        xt -- input data at timestep "t", numpy array of shape (hidden_dim, batch).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (hidden_dim, batch)
        c_prev -- Memory state at timestep "t-1", numpy array of shape (hidden_dim, batch)
        '''

        # Concatenate a_prev and xt 
        concat = np.zeros((self.hidden_dim + self.input_dim, self.batch))
        concat[: self.hidden_dim, :] = a_prev
        concat[self.hidden_dim :, :] = xt

        # Compute values for ft, it, cct, c_next, ot, a_next 
        ft = sigmoid(np.dot(self.Wf, concat) + self.bf)
        it = sigmoid(np.dot(self.Wi, concat) + self.bi)
        cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
        c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
        ot = sigmoid(np.dot(self.Wo, concat) + self.bo)
        a_next = np.multiply(ot, np.tanh(c_next))

        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt)

        return a_next, c_next, cache

    def lstm_forward(self, x, a0):
        '''
        x -- Input data for every time-step, of shape (input_dim, batch, len).
        a0 -- Initial hidden state, of shape (hidden_dim, batch)
        Returns:
        a -- Hidden states for every time-step, numpy array of shape (hidden_dim, batch, len)
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
        return:
            yt_pred -- prediction at last timestep, numpy array of shape (classes, batch)
        '''
        caches = []

        len = x.shape[2]

        a = np.zeros((self.hidden_dim, self.batch, len))
        c = np.zeros((self.hidden_dim, self.batch, len))

        # Initialize a_next and c_next
        a_next = a0
        c_next = np.zeros((self.hidden_dim, self.batch))

        # loop over all time-steps
        for t in range(len):
            # Update next hidden state, next memory state, compute the prediction, get the cache
            a_next, c_next, cache = self.lstm_cell_forward(xt=x[:, :, t], a_prev=a_next, c_prev=c_next)
            # Save the value of the new "next" hidden state in a 
            a[:, :, t] = a_next
            # Save the value of the next cell state
            c[:, :, t] = c_next
            # Append the cache into caches
            caches.append(cache)

        # store values needed for backward propagation in cache
        caches = (caches, x)

        # Compute prediction of the last LSTM cell 
        yt_pred = softmax(np.dot(self.Wy, a_next) + self.by)

        return a, c, caches, yt_pred
    
    def lstm_cell_backward(self, da_next, dc_next, cache):
        '''
        Arguments:
        da_next -- Gradients of next hidden state, of shape (hidden, batch)
        dc_next -- Gradients of next cell state, of shape (hidden, batch)
        cache -- cache storing information from the forward pass
        Returns:
        gradients -- python dictionary containing:
            dxt -- Gradient of input data at time-step t, of shape (input_dim, batch)
            da_prev -- Gradient of the previous hidden state, numpy array of shape (hidden_dim, batch)
            dc_prev -- Gradient of the previous memory state, of shape (hidden_dim, batch, len)
            dWf -- Gradient of the weight matrix of the forget gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
            dWi -- Gradient of the weight matrix of the update gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
            dWc -- Gradient of the weight matrix of the memory gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
            dWo -- Gradient of the weight matrix of the output gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
            dbf -- Gradient of biases of the forget gate, of shape (hidden_dim, 1)
            dbi -- Gradient of biases of the update gate, of shape (hidden_dim, 1)
            dbc -- Gradient of biases of the memory gate, of shape (hidden_dim, 1)
            dbo -- Gradient of biases of the output gate, of shape (hidden_dim, 1)
        '''

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt) = cache

        # Compute gates related derivatives
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

        # Compute parameters related derivatives.
        concat = np.zeros((self.hidden_dim + self.input_dim, self.batch))
        concat[: self.hidden_dim, :] = a_prev
        concat[self.hidden_dim :, :] = xt
        concat = concat.T
        #concat = np.vstack((a_prev, xt)).T
        dWf = np.dot(dft, concat)
        dWi = np.dot(dit, concat)
        dWc = np.dot(dcct, concat)
        dWo = np.dot(dot, concat)
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Compute derivatives w.r.t previous hidden state, previous memory state and input.
        da_prev = np.dot(self.Wf[:, :self.hidden_dim].T, dft) + np.dot(self.Wi[:, :self.hidden_dim].T, dit) + np.dot(self.Wc[:, :self.hidden_dim].T, dcct) + np.dot(self.Wo[:, :self.hidden_dim].T, dot)
        dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = np.dot(self.Wf[:, self.hidden_dim:].T, dft) + np.dot(self.Wi[:, self.hidden_dim:].T, dit) + np.dot(self.Wc[:, self.hidden_dim:].T, dcct) + np.dot(self.Wo[:, self.hidden_dim:].T, dot)

        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_next": da_prev, "dc_next": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

        return gradients

    def lstm_backward(self, loss, caches):
        '''
        Arguments:
        loss -- loss from CrossEntropy, of shape (classes, batch)
        caches -- cache storing information from the forward pass (lstm_forward)
        returns:
        gradients -- python dictionary containing:
                dx -- Gradient of inputs, of shape (input_dim, batch, eln)
                da0 -- Gradient of the previous hidden state, numpy array of shape (hidden_dim, batch)
                dWf -- Gradient of the weight matrix of the forget gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
                dWi -- Gradient of the weight matrix of the update gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
                dWc -- Gradient of the weight matrix of the memory gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
                dWo -- Gradient of the weight matrix of the save gate, numpy array of shape (hidden_dim, hidden_dim + input_dim)
                dbf -- Gradient of biases of the forget gate, of shape (hidden_dim, 1)
                dbi -- Gradient of biases of the update gate, of shape (hidden_dim, 1)
                dbc -- Gradient of biases of the memory gate, of shape (hidden_dim, 1)
                dbo -- Gradient of biases of the save gate, of shape (hidden_dim, 1)
        '''
        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches

        len = x.shape[2]
        # initialize the gradients with the right sizes (≈12 lines)
        dx = np.zeros((self.input_dim, self.batch, len))
        da0 = np.zeros((self.hidden_dim, self.batch))
        dc_next = np.zeros((self.hidden_dim, self.batch))
        dWf = np.zeros((self.hidden_dim, self.hidden_dim + self.input_dim))
        dWi = np.zeros((self.hidden_dim, self.hidden_dim + self.input_dim))
        dWc = np.zeros((self.hidden_dim, self.hidden_dim + self.input_dim))
        dWo = np.zeros((self.hidden_dim, self.hidden_dim + self.input_dim))
        dbf = np.zeros((self.hidden_dim, 1))
        dbi = np.zeros((self.hidden_dim, 1))
        dbc = np.zeros((self.hidden_dim, 1))
        dbo = np.zeros((self.hidden_dim, 1))


        (a_last, _, _, _, _, _, _, _, _) = caches[len-1]

        # gradient from softmax
        dz = np.zeros((self.classes,self.batch))
        df = softmax_derivate(np.dot(self.Wy, a_last))
        for i in range(0,self.batch):
            #print( dz[:,i].shape,loss[:,i].shape,df[:,:,i].shape)
            dz[:,i] = loss[:,i].dot(df[:,:,i])

        dWy = np.dot(dz, a_last.T)
        dby = np.sum(dz, axis=1, keepdims=True)
        da_next = np.dot(self.Wy.T, dz)

        # loop back over the whole sequence
        for t in reversed(range(len)):
            # Compute all gradients using lstm_cell_backward
            gradients = self.lstm_cell_backward(da_next=da_next, dc_next=dc_next, cache=caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            dx[:, :, t] = gradients["dxt"]
            dWf = dWf+gradients["dWf"]
            dWi = dWi+gradients["dWi"]
            dWc = dWc+gradients["dWc"]
            dWo = dWo+gradients["dWo"]
            dbf = dbf+gradients["dbf"]
            dbi = dbi+gradients["dbi"]
            dbc = dbc+gradients["dbc"]
            dbo = dbo+gradients["dbo"]
            da_next = gradients['da_next']
            dc_next = gradients['dc_next']

        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients['da_next']

        gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby":dby}

        return gradients
    

    def update_parameters(self, gradients):
        """
        梯度下降
        :param gradients:
        :return:
        """
        self.Wf += -self.alpha * gradients["dWf"]
        self.Wi += -self.alpha * gradients["dWi"]
        self.Wc += -self.alpha * gradients['dWc']
        self.Wo += -self.alpha * gradients["dWo"]
        self.Wy += -self.alpha * gradients['dWy']

        self.bf += -self.alpha * gradients['dbf']
        self.bi += -self.alpha * gradients['dbi']
        self.bc += -self.alpha * gradients['dbc']
        self.bo += -self.alpha * gradients['dbo']
        self.by += -self.alpha * gradients['dby']
    
if __name__ == '__main__':
    # a = np.array([[1,2],[1,2]])
    # print(softmax_derivate(a))
    loss = np.array([[2,2],[1,1]])
    dz = np.zeros((2,2))
    for i in range(0,2):
        df = softmax_derivate(np.array([[1,0],[0,1]]))
        print(df)
        dz[:,i] = loss[:,i].dot(df[:,:,i])
    print(dz)