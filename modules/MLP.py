import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    eps = 1e-8
    out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)


class MLP():
    def __init__(self, features, labels):
        self.D_in, self.H1, self.H2, self.D_out = features, 100, 50, labels
        self.epochs, self.batch_size = 200, 32
        self.learning_rate = 1e-2

        # Произвольно инициализируем веса
        self.w1 = np.random.randn(self.D_in, self.H1)
        self.w2 = np.random.randn(self.H1, self.H2)
        self.w3 = np.random.randn(self.H2, self.D_out)

        self.b1 = np.random.randn(1, self.H1)
        self.b2 = np.random.randn(1, self.H2)
        self.b3 = np.random.randn(1, self.D_out)

    def predict(self, x):
        a1 = sigmoid(x.dot(self.w1) + self.b1)
        a2 = sigmoid(a1.dot(self.w2) + self.b2)
        return softmax(a2.dot(self.w3) + self.b3)

    def fit(self, x_train, labels):
        train_num = x_train.shape[0]
        bvec = np.ones((1, self.batch_size))

        y_train = np.zeros((train_num, self.D_out))
        y_train[np.arange(train_num), labels] = 1

        for epoch in range(self.epochs):
            permut = np.random.permutation(
                train_num // self.batch_size * self.batch_size).reshape(-1, self.batch_size)
            for b_idx in range(permut.shape[0]):
                x, y = x_train[permut[b_idx, :]], y_train[permut[b_idx, :]]

                # Вычисляем прогнозируемое значение y
                a1 = sigmoid(x.dot(self.w1) + self.b1)
                a2 = sigmoid(a1.dot(self.w2) + self.b2)
                out = softmax(a2.dot(self.w3) + self.b3)

                # Считаем градиенты весов
                grad_out = out - y
                grad_w3 = a2.T.dot(grad_out)

                grad_a2 = grad_out.dot(self.w3.T)
                grad_a2 = np.multiply(grad_a2, (a2 - np.square(a2)))
                grad_w2 = a1.T.dot(grad_a2)

                grad_a1 = grad_a2.dot(self.w2.T)
                grad_a1 = np.multiply(grad_a1, (a1 - np.square(a1)))
                grad_w1 = x.T.dot(grad_a1)

                # Обновляем значения весов
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * bvec.dot(grad_a1)
                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * bvec.dot(grad_a2)
                self.w3 -= self.learning_rate * grad_w3
                self.b3 -= self.learning_rate * bvec.dot(grad_out)
