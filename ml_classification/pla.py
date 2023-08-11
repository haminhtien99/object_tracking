import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, x, y, lr = 1, lamda = 0):
        self.length = len(x)
        self.x = x
        self.y = y
        self.lr = lr
        self.lamda = lamda
        self.w0 = 0
        self.w = np.zeros(len(self.x[0]))

    def predict(self, w, w0, x):
        y_predict = np.dot(x, w) + w0
        if y_predict >= 0: y_predict = 1
        else: y_predict = -1
        return y_predict

    def update(self, y_i, x_i):
        self.w += self.lr*y_i*x_i
        self.w0 += self.lr*y_i

    def train(self, epochs = 100):
        Finish = False
        count = 0
        while not Finish:
            count += 1
            err = 0
            yy = 0
            xx = np.zeros(len(self.x[0]))
            for i in range(self.length):
                x_i = self.x[i]
                y_i = self.y[i]
                # print(y_i, self.predict(self.w, self.w0, x_i), count - epochs)

                if self.predict(self.w, self.w0, x_i)!= y_i :
                    err += 1
                    yy = y_i
                    xx = x_i
            # Update last element
            self.update(yy, xx)

            if err == 0 or count >= epochs: Finish = True
            # print(self.w, self.w0)

        return self.w, self.w0


    def eval(self,X_test, y_test):
        count = 0
        samples = len(y_test)
        y_predict = np.zeros(len(y_test))
        for i in range(len(X_test)):
            x_i = X_test[i]
            y_i = y_test[i]
            y_predict[i] = self.predict(self.w, self.w0, x_i)
            if y_i * y_predict[i] == 1: count += 1
        score = count/samples * 100
        return y_predict, score
    def show_graph(self):
        cols = ['blue', 'red']

        # построение точек
        for k in np.unique(self.y):
            if k == -1: col = cols[0]
            else: col = cols[1]
            plt.plot(self.x[self.y==k,0], self.x[self.y==k,1], 'o', label='класс {}'.format(k), color=col)

        if self.w[1] != 0:
            b = - self.w0/self.w[1]
            k = - self.w[0]/self.w[1]
        else:
            pass
        x_axis = np.linspace(-1, 1)
        y_axis = k* x_axis + b
        plt.plot(x_axis, y_axis, linewidth=2)
        plt.legend(loc='best')
        plt.show()
np.random.seed(0)
l = 500
n = 2
X1 = np.array([[-1,-1]]) + 0.5*np.random.randn(l, n)
X2 = np.array([[1,1]]) + 0.5*np.random.randn(l, n)

X = np.vstack([X1, X2])
y = np.hstack([[-1]*l, [1]*l])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

model = Perceptron(X_train, y_train, lr = 0.5)
model.train(epochs = 1000)
model.show_graph()