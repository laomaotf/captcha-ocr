#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
统计样本的标签，并写入文件labels.json
"""
import json,os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']

csv_path = '../sample/train.csv' #filename, label
image_folder = '../sample/train/'
image_num = 0
max_label_len = 0
labels = set()
labels_count = {}
with open(csv_path,'r',encoding='utf8') as f:
    f.readline()
    for line in f:
        split_result = line.strip().split(',')
        if len(split_result) == 2:
            filename, label = split_result
            if not os.path.exists(os.path.join(image_folder,filename)):
                continue
            image_num += 1
            if label:
                if len(label) > max_label_len:
                    max_label_len = len(label)
                for word in label:
                    #if word == '"':
                    #    print(filename)
                    labels.add(word)
                    if word not in labels_count:
                        labels_count[word] = 1
                    else:
                        labels_count[word] += 1
        else:
            pass

total = sum([labels_count[key] for key in labels_count])
items = sorted(labels_count.items(), key = lambda d: d[1])
x = [k[0] for k in items]
y = [k[1] for k in items]

print('出现频次最高: {} {}'.format(x[0],y[0]))
print('出现频次最低: {} {}'.format(x[-1],y[-1]))

bins = [10 * k + 10 for k in range(10)]
figs,axes = plt.subplots(nrows=2,ncols=1)
axes[0].hist(y,bins=bins)
axes[1].plot(x,y)

print("扫描图像{}张".format(image_num))
print("共有标签{}种".format(len(labels)))
print("最大标签长度{}".format(max_label_len))
with open("./labels.json", "w") as f:
    labels = sorted(list(labels))
    f.write(json.dumps("".join(list(labels)), ensure_ascii=False))
print("将标签列表写入文件labels.json成功")
plt.show()