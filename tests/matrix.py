#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
@Time : 2021/7/8 10:33
@Author : 詹荣瑞
@File : matrix.py
@desc : 本代码未经授权禁止商用
"""
import numpy as np
import boning as bn
from boning.variable import Variable

A = Variable([[2, 2], [1, 3]], require_grad=True)
y = Variable([[1], [2]], require_grad=True)
# 向量四则运算
u = A @ y
v = u.T @ u
print(v)
v.backward()
print(A.grad, y.grad)
print(2 * A @ y @ y.T, 2 * A.T @ A @ y)
