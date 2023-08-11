import math
from base64 import b16encode
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Построить выпуклую оболочку из множеста точек на плоскости по алгоритму Jarvis
def angle(A, B):
    y = B[1] - A[1]
    x = B[0] - A[0]
    if x == 0:
        ang = math.pi / 2
    elif x > 0: ang = math.atan(y/x)
    else: ang = math.atan(y/x) + math.pi

    # -Pi/2 <ang < 3Pi/2
    if ang <0 : ang += 2* math.pi #  0 < ang < 2Pi

    return ang

def rotation_angle(p_prev, p_now, P):
    ang1 = angle(p_prev, p_now)
    ang2 = angle(p_now, P)
    if ang2 >= ang1: aa = ang2 - ang1
    else: aa = ang2 + 2*math.pi-ang1
    return aa

def Jarvis(X):
    Indexs = list(range(len(X)))
    P = []

    p0 = X[0]
    for A in X:
        if p0[1] > A[1] or (p0[1] == A[1] and p0[0] < A[0]): p0 = A
    P.append(p0)

    if X[0][0] != p0[0]:
        p1 = X[0]
        ind = 0
    else:
        p1 = X[1]
        ind = 1

    # Find p1
    min = rotation_angle([p0[0] - 1,  p0[1]], p0, p1)
    for i in Indexs:
        if i == ind : continue
        A = X[i]
        roto = rotation_angle([p0[0] - 1,  p0[1]], p0, A)
        if min > roto:
            min = roto
            p1 = A
            ind = i
    P.append(p1)
    Indexs.remove(ind)

    Finish = False
    while Finish == False:
        p_now = P[-1]
        p_prev = P[-2]
        ind = Indexs[0]
        p_next=X[ind]
        min = rotation_angle(p_prev, p_now, p_next)

        for i in Indexs:
            if i == ind: continue
            A = X[i]
            roto = rotation_angle(p_prev, p_now, A)
            if min > roto:
                min = roto
                p_next = A
                ind = i
            if min == roto :
                len1 = math.sqrt((A[0]- p_now[0])* (A[0]- p_now[0]) + (A[1]- p_now[1])* (A[1]- p_now[1]))
                len2 = math.sqrt((p_next[0]- p_now[0])* (p_next[0]- p_now[0]) + (p_next[1]- p_now[1])* (p_next[1]- p_now[1]))
                if len1 > len2:
                    ind  = i
                    p_next = A
        Indexs.remove(ind)
        P.append(p_next)
        # print(P, Indexs)

        if p_next == p0: Finish = True
    return P
# Построить прямую из двух точек
def linear(point1, point2):
    if point1[0] - point2[0] != 0:
        a1 = -(point1[1] - point2[1])/(point1[0]- point2[0])
        a2 = 1
        b = a1 * point1[0] + a2*point1[1]
    else:
        a1 = 1
        a2 = 0
        b = point1[0]
    return a1, a2, b

def hull(P):
    A =[]
    bb =[]
    for i in range(len(P)-1):
        a1, a2, b = linear(P[i], P[i+1])
        for p in P:
            conf = a1*p[0] + a2*p[1]
            if conf == b: continue
            elif conf < b: break
            else:
                a1 = -a1
                a2 = -a2
                b = -b
                break
        A.append([a1,a2])
        bb.append(b)
    return A, bb
# Найти минимальное расстояние между двух выпуклой оболочки (с использованием библиотеки CVXOPT)
# Для задачи SVM нужна оптимальная разделяющая гиперплоскость( с максимальным margin). margin это найденное минимальное расстояние. 


def sign(a):
    res = -1
    if a >= 0: res = 1
    return res
class SVM:
    def __init__(self, x, y): # x, y - np.array
        self.x = x
        self.y = y
        self.w = np.zeros(len(self.x[0]))
        self.w0 = 0
        self.margin = 0
        self.supportPoint1 = np.zeros(len(self.x[0]))
        self.supportPoint2 = np.zeros(len(self.x[0]))
        self.length = len(x)
        self.P = None
        
    def fit(self):
        cls1 = []
        cls2 = []
        for i in range(self.length):
            if self.y[i] == 1: cls1.append(self.x[i].tolist())
            else: cls2.append(self.x[i].tolist())
        P1 =Jarvis(cls1)
        P2 = Jarvis(cls2)
        
        A1, b1  = hull(P1)
        A2, b2 = hull(P2)
        
        # use matrix for CVXOPT
        I = matrix([[1., 0.], [0., 1.]])

        AA1 = matrix(np.array(A1))
        AA2 = matrix(np.array(A2))
        bb1 = matrix(b1)
        bb2 = matrix(b2)
        
        solvers.options['show_progress'] = False
        p0 = P1[0]
        argPoint = matrix(p0)
        sol = solvers.qp(I, -matrix(p0), AA2, bb2)
        argX = sol['x'] # x- matrix 2x1
        vectorR = argX - matrix(p0)
        margin = math.sqrt(vectorR[0]*vectorR[0] + vectorR[1]*vectorR[1])

        for p in P1:
            if p0 == p: continue
            sol = solvers.qp(I, -matrix(p), AA2, bb2)
            x = sol['x'] # x- matrix 2x1
            vectorR = x - matrix(p)
            R =math.sqrt(vectorR[0]*vectorR[0] + vectorR[1]*vectorR[1])
            if R < margin:
                margin = R
                argX = x
                argPoint = matrix(p)

        for p in P2:
            sol = solvers.qp(I, -matrix(p), AA1, bb1)
            x = sol['x'] # x- matrix 2x1
            vectorR = x - matrix(p)
            R =math.sqrt(vectorR[0]*vectorR[0] + vectorR[1]*vectorR[1])
            if R < margin:
                margin = R
                argX = x
                argPoint = matrix(p)

        M = (argX + argPoint)/2
        
        self.w = np.array(argPoint - argX).transpose()[0]
        self.w0 = self.w.dot(np.array(M))[0]
        self.supportPoint1 = np.array(argPoint).transpose()[0]
        self.supportPoint2 = np.array(argX).transpose()[0]
        self.P = [P1, P2]
        
        return self.margin, self.w, self.w0
    def predict(self, X, Y):
        count = 0
        Y_pred = []
        for i in range(len(X)):
            y_pred= sign(self.w.dot(X[i]) - self.w0)
            Y_pred.append(y_pred)
            if y_pred != Y[i]: count += 1
        err = count/len(X)
        return Y_pred, err
    
    def vizualization(self):
        cols = ['blue', 'red']
        x_values = np.linspace(min(self.x[:,0]), max(self.x[:, 0]), 100)
        
        if self.w[1] != 0:
            y0_values = self.w0/self.w[1] * np.ones(100)- self.w[0]/self.w[1] * x_values
            b1 = self.w.dot(self.supportPoint1)
            b2 = self.w.dot(self.supportPoint2)
            y1_values = b1/self.w[1] * np.ones(100)- self.w[0]/self.w[1] * x_values
            y2_values = b2/self.w[1] * np.ones(100)- self.w[0]/self.w[1] * x_values
            

        # построение точек
        for k in np.unique(self.y):
            if k == -1: col = cols[0]
            else: col = cols[1]
            plt.plot(self.x[self.y==k,0], self.x[self.y==k,1], 'o', label='класс {}'.format(k), color=col)
        for p in self.P[0]:
            plt.plot(p[0], p[1], 'o',color='green')

        for p in self.P[1]:
            plt.plot(p[0], p[1], 'o', color='black')
            
        plt.plot(x_values, y0_values)
        plt.plot(x_values, y1_values)
        plt.plot(x_values, y2_values)
        
        plt.plot(self.supportPoint1[0], self.supportPoint1[1], 'x', color = 'orange')
        plt.plot(self.supportPoint2[0], self.supportPoint2[1], 'x', color = 'orange')
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

model = SVM(X,y)
model.fit()
_, err = model.predict(X_test, y_test)
print(err)
model.vizualization()