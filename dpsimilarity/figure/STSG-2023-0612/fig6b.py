import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局字体
mpl.rcParams['font.family'] = 'times new roman'
mpl.rcParams['font.size'] = 14

# 定义方法和属性值
methods = ['STSG', 'BERT', 'DistilBERT', 'T5', 'SpanBERT']
pearson_values = [0.61, 0.98, 1.11, 0.68, 1.11]
p_values = [0.56, 1.61, 1.68, 0.87, 2.10]

# 绘制柱状图
fig, ax = plt.subplots()

width = 0.4
ax.bar(methods, pearson_values, width=width, label='MAE', color='#FF7373',  edgecolor="k", linewidth=1)
ax.bar([i + width for i in range(len(methods))], p_values, width=width, label='MSE', color='#FFDC73', edgecolor="k", linewidth=1)
ax.set_xlabel('Methods')
ax.set_ylabel('Error')
# ax.set_title('BIOSSES by Training Ratio 100%')
ax.legend()
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# 显示y轴网格线
# ax.yaxis.grid(color='lightgray', alpha=0.3, zorder=0)

# 在柱状图上显示数值
for i in range(len(methods)):
    ax.text(i, pearson_values[i]+0.02, str(pearson_values[i]), ha='center')
    ax.text(i+width, p_values[i]+0.02, str(p_values[i]), ha='center')

#移动x轴标签的位置
ax.set_ylim(0, 1.35 * max(pearson_values+p_values))

ax.set_xticks([i + width / 2 for i in range(len(methods))])
# ax.set_xticklabels(methods, rotation=45, ha='right')
plt.savefig("img6b.png", dpi=800)
plt.show()

