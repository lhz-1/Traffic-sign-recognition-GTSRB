#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


# 在保证初始代码结构不变的情况下，可以通过下面“char_count”一个函数实现题目的要求，也可以通过多个函数实现

# =============这里往下是你主要编写代码的地方，此区域外的代码都不能删除==================
# 入口函数，不能删除，因为在下方有调用

def char_count(str1):
    c= []
    count = 1
    a = len(str1)
    b = list(str1)
    for i in range(1,a):
        if b[i-1] == b[i]:
            count += 1
        else:
            c.append(count)
            c.append(b[i-1])
            count = 1

    return c

# 其他函数可以从这里写起

# =============这里往上是你主要编写代码的地方，此区域外的代码都不能删除==================


try:
    v = input()
except:
    v = None

res = char_count(v)

for res_cur in res:
    print(str(res_cur) ,end = ' ')