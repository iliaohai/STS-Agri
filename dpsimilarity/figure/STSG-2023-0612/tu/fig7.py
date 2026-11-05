import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['font.family'] = 'times new roman'
mpl.rcParams['font.size'] = 8

# 创建一个大图和两个子图
fig, axes = plt.subplots(nrows=1, ncols=2)

# 加载并显示第一个图像文件
img1 = mpimg.imread('img6a.png')
axes[0].imshow(img1)
axes[0].axis('off')  # 关闭坐标轴
axes[0].text(0.5, -0.05, '(a) BIOSSES on training ratio of 100%', transform=axes[0].transAxes,
             ha='center', va='center')

# 加载并显示第二个图像文件
img2 = mpimg.imread('img6b.png')
axes[1].imshow(img2)
axes[1].axis('off')  # 关闭坐标轴
axes[1].text(0.5, -0.05, '(b) BIOSSES on training ratio of 50%', transform=axes[1].transAxes,
             ha='center', va='center')

# 调整子图之间的间距和边距
plt.subplots_adjust(wspace=0, left=0, right=1, bottom=0, top=1)

# 显示大图
fig.savefig('fig7.png', dpi=600)
plt.show()
