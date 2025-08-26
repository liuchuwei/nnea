#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肿瘤药物数据处理脚本
将R脚本转换为Python版本，处理CTR数据库中的肿瘤药物数据
为每个药物创建nadata对象，保存到datasets/tumor_drug文件夹中
"""


import pandas as pd
from nnea.io._nadata import nadata

exp = pd.read_csv("tmp/exp.txt")
phe = pd.read_csv("tmp/phe.txt")

X = exp.iloc[:,1:].T.values

# 过滤表型数据并确保顺序一致
Meta = phe

# 过滤基因数据
Var = exp.iloc[:,0]

# 创建nadata对象
nadata_obj = nadata(
    X=X,
    Meta=Meta,
    Var=Var,
    Prior=None
)


nadata_obj.save("datasets/tmp.pkl", format='pickle', save_data=True)

