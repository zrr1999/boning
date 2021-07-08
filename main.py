import numpy as np
import boning as bn
from boning.variable import Variable

x = Variable([[2], [1]], require_grad=True)
y = Variable([[1], [2]], require_grad=True)
# 向量四则运算
t = (x * y).T
v = (x * y).T @ (1 / x)
w = (x * y).T @ y
u = (x * y).T @ (y + 1 / x + 1)

t.backward()
print(x.grad)
t.zero_grad()

v.backward()
print(x.grad)
v.zero_grad()

w.backward()
print(x.grad)
w.zero_grad()

u.backward()
print(x.grad)
u.zero_grad()


# sigmoid函数
from boning.utils import sigmoid


u = sigmoid(x)
loss = u.T @ u
loss.backward()
print(x.grad)
loss.zero_grad()


# # relu函数
# def relu(a):
#     return np.maximum(0, a)
#
#
# u = relu(x)
# loss = u.T @ u
# loss.backward()
# print(x.grad)
# loss.zero_grad()
