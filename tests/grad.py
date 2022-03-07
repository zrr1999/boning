#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/2/11 15:58
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
from boning import Variable

a =Variable([[[1],[2]]])
b =Variable([[[1,1]]])

print(a.shape,b.shape)
print(a@b)
