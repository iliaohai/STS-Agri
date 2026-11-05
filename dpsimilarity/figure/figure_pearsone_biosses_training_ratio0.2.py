import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f1_1 = [0.509, 0.416, 0.418, 0.456, 0.453]
f1_2 = [0.044, 0.109, 0.107, 0.076, 0.078]


x = ['DPSimilarity', 'BERT', 'DistilBERT', 'T5', 'SpanBERT']
x_len = np.arange(len(x))
total_width, n = 0.9, 3
width = 0.3
xticks = x_len - (total_width - width) / 2
plt.figure(figsize=(17, 12), dpi=200)

ax = plt.axes()
plt.grid(axis="y", c='#d2c9eb', linestyle='--', zorder=0)

plt.bar(xticks, f1_1, width=0.9 * width, label="Pearson", color="#92a6be", edgecolor='black', linewidth=2,zorder=10)
plt.bar(xticks + width, f1_2, width=0.9 * width, label="P-value", color="#c48d60", edgecolor='black', linewidth=2,zorder=10)

plt.text(xticks[0], f1_1[0] + 0.3, f1_1[0], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
plt.text(xticks[1], f1_1[1] + 0.3, f1_1[1], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
plt.text(xticks[2], f1_1[2] + 0.3, f1_1[2], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
plt.text(xticks[3], f1_1[3] + 0.3, f1_1[3], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
plt.text(xticks[4], f1_1[4] + 0.3, f1_1[4], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)

plt.text(xticks[0] + width, f1_2[0] + 0.3, f1_2[0], ha='center', fontproperties='Times New Roman', fontsize=35,zorder=10)
plt.text(xticks[1] + width, f1_2[1] + 0.3, f1_2[1], ha='center', fontproperties='Times New Roman', fontsize=35,zorder=10)
plt.text(xticks[2] + width, f1_2[2] + 0.3, f1_2[2], ha='center', fontproperties='Times New Roman', fontsize=35,zorder=10)
plt.text(xticks[3] + width, f1_2[3] + 0.3, f1_2[3], ha='center', fontproperties='Times New Roman', fontsize=35,zorder=10)
plt.text(xticks[4] + width, f1_2[4] + 0.3, f1_2[4], ha='center', fontproperties='Times New Roman', fontsize=35,zorder=10)


plt.legend(prop={'family': 'Times New Roman', 'size': 35}, ncol=2)
x_len = [-0.1, 0.9, 1.9,2.9,3.9]
x_len = np.array(x_len)
plt.xticks(x_len, x, fontproperties='Times New Roman', fontsize=40)
plt.yticks(fontproperties='Times New Roman', fontsize=40)
plt.ylim(ymin=0.01)
plt.xlabel("Methods", fontproperties='Times New Roman', fontsize=45)
plt.ylabel("Pearsone (%)", fontproperties='Times New Roman', fontsize=45)


ax.spines['bottom'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['right'].set_color('black')
ax.spines['left'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['left'].set_color('black')

plt.show()
