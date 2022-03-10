import pickle
import numpy as np
import scipy.special

def relu(x, d=False):
    if d:
        x[x >= 0] = 1
    else:
        x[x < 0] = 0
    return x


def softmax(x, d=False):
    """Compute the softmax of vector x in a numerically stable way."""
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# def softmax(z, d=False):
#     return np.exp(z) / np.sum(np.exp(z))

def sigmoid(z, d=False):
    s = scipy.special.expit(z)
    if d:
        return z * s * (1 - s)
    return z

class Layer:
    def __init__(self, number_of_features, activation_units,
                 layer_type, activation_function=sigmoid, rate=0.1):
        self.weight = np.random.rand(activation_units, number_of_features) * 0.01
        #xavier initialisation
        # self.weight = np.random.rand(activation_units, number_of_features) * np.sqrt(1.0/activation_units)
        self.bias = np.zeros((activation_units, 1))
        self.activation_function = activation_function
        self.type = layer_type
        self.rate = rate

        self.vdW = np.zeros(self.weight.shape)
        self.vdb = np.zeros(self.bias.shape)
        self.sdW = np.zeros(self.weight.shape)
        self.sdb = np.zeros(self.bias.shape)

        assert(self.weight.shape == (activation_units, number_of_features))
        assert(self.bias.shape == (activation_units, 1))

    def linear_forward(self, x):
        self.z = (self.weight @ x) + self.bias
        assert(self.z.shape == (self.weight.shape[0], x.shape[1]))
        return self.z

    def linear_activation_forward(self, A_prev):
        self.A_prev = A_prev
        self.A = self.activation_function(self.linear_forward(A_prev))
        assert (self.A.shape == (self.weight.shape[0], A_prev.shape[1]))
        return self.A

    def linear_backward(self, A_prev):
        dZ = self.activation_function(A_prev, d=True)
        m = A_prev.shape[0]
        self.dW = np.dot(dZ, self.A_prev.T) / m
        self.db = np.sum(np.matrix(dZ), axis=1) / m
        dA_prev = np.dot(self.weight.T, dZ)
        assert(dA_prev.shape == self.A_prev.shape)
        assert(self.dW.shape == self.weight.shape)
        assert (self.db.shape == self.bias.shape)
        return dA_prev

    # def update_parameters(self):
    #     self.weight -= self.rate * self.dW
    #     self.bias -= self.rate * self.db

    def update_parameters(self, n, beta1=0.9, beta2=0.999):
        epsilon=10e-8
        vdw = self.vdW /(1-np.power(beta1, n))
        sdw = self.sdW /(1-np.power(beta2, n))

        vdb = self.vdb /(1-np.power(beta1, n))
        sdb = self.sdb /(1-np.power(beta2, n))

        self.weight -= self.rate * (vdw / (np.sqrt(sdw) + epsilon))
        self.bias -= self.rate * (vdb / (np.sqrt(sdb) + epsilon))

    def adam(self, beta1=0.9, beta2=0.999):
        self.vdW = beta1 * self.vdW + (1 - beta1) * self.dW
        self.sdW = beta1 * self.sdW + (1 - beta2) * (self.dW**2)

        self.vdb = beta1 * self.vdb + (1 - beta1) * self.db
        self.sdb = beta1 * self.sdb + (1 - beta2) * (np.power(self.db, 2))


class MultilayerPerceptron:
    def __init__(self, layers, lr=0.001, path="mp.pickle"):
        self.lr = lr
        self.layers = layers
        self.path = path

    def loss(self, p, y, eps=1e-15):
        m = y.shape[0]
        return np.squeeze(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) / -m)

    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.linear_activation_forward(x)
        return x

    def backward_propagation(self, X, Y, n):
        AL = self.forward_propagation(X)
        Y = Y.reshape(AL.shape)
        A_prev = AL-Y
        for i in range(len(self.layers)-1, -1, -1):
            A_prev = self.layers[i].linear_backward(A_prev)
            self.layers[i].adam()
            self.layers[i].update_parameters(n)
        return AL

    def load(self):
        with open(self.path, 'rb') as handle:
            self = pickle.load(handle)
        return self

    def save(self):
        with open(self.path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

