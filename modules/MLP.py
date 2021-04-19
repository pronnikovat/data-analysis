import numpy as np


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class Relu:
    @staticmethod
    def activation(z):
        return np.maximum(z, 0)

    @staticmethod
    def prime(z):
        return np.where(Relu.activation(z) < 0, 0, 1)


class MSE:
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true


class MLP:
    def __init__(self, layers, activation_functions):
        self.layers = layers
        self.n_layers = len(layers)
        self.loss_func = None
        self.learning_rate = None
        self.activ_func = activation_functions
        self.w = {}
        self.b = {}
        self.activation_functions = {}
        for i in range(len(layers) - 1):
            self.w[i + 1] = np.random.randn(layers[i], layers[i + 1]) / np.sqrt(layers[i])
            self.b[i + 1] = np.zeros(layers[i + 1])
            self.activation_functions[i + 2] = activation_functions[i]

    def feed_forward(self, X):
        z = {}
        a = {1: X}
        for i in range(1, self.n_layers):
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activation_functions[i + 1].activation(z[i + 1])
        return z, a

    def predict(self, X):
        _, a = self.feed_forward(X)
        return a[self.n_layers]

    def back_prop(self, z, a, y):
        back_prop_error = self.loss_func.prime(y, a[self.n_layers]) * self.activ_func[-1].prime(a[self.n_layers])
        dC_dw = np.dot(a[self.n_layers - 1].T, back_prop_error)
        update_parameters = {
            self.n_layers - 1: (dC_dw, back_prop_error)
        }
        for n in reversed(range(2, self.n_layers)):
            back_prop_error = np.dot(back_prop_error, self.w[n].T) * self.activation_functions[n].prime(z[n])
            dC_dw = np.dot(a[n - 1].T, back_prop_error)
            update_parameters[n - 1] = (dC_dw, back_prop_error)
        for i, update_param in update_parameters.items():
            self.w[i] -= self.learning_rate * update_param[0]
            self.b[i] -= self.learning_rate * np.mean(update_param[1], 0)

    def fit(self, X_train, y_train, loss_func=MSE, epochs=10, batch_size=64, learning_rate=0.01):
        self.loss_func = loss_func(self.activation_functions[self.n_layers])
        self.learning_rate = learning_rate
        for i in range(epochs):
            seed = np.arange(X_train.shape[0])
            np.random.shuffle(seed)
            x_ex = X_train[seed]
            y_ex = y_train[seed]
            for j in range(X_train.shape[0] // batch_size):
                m = j * batch_size
                n = (j + 1) * batch_size
                z, a = self.feed_forward(x_ex[m:n])
                self.back_prop(z, a, y_ex[m:n])
