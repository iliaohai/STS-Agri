import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['font.family'] = 'times new roman'
mpl.rcParams['font.size'] = 8

# 创建一个大图和四个子图
fig, axes = plt.subplots(nrows=2, ncols=2)

# 加载并显示第一个图像文件
img1 = mpimg.imread('img5a.png')
axes[0, 0].imshow(img1)
axes[0, 0].axis('off')  # 关闭坐标轴
axes[0, 0].text(0.5, -0.04, '(a) BIOSSES on training ratio of 100%', transform=axes[0, 0].transAxes,
                ha='center', va='center')

# 加载并显示第二个图像文件
img2 = mpimg.imread('img5b.png')
axes[0, 1].imshow(img2)
axes[0, 1].axis('off')  # 关闭坐标轴
axes[0, 1].text(0.5, -0.04, '(b) BIOSSES on training ratio of 50%', transform=axes[0, 1].transAxes,
                ha='center', va='center')

# 加载并显示第三个图像文件
img3 = mpimg.imread('img5c.png')
axes[1, 0].imshow(img3)
axes[1, 0].axis('off')  # 关闭坐标轴
axes[1, 0].text(0.5, -0.04, '(c) BIOSSES on training ratio of 20%', transform=axes[1, 0].transAxes,
                ha='center', va='center')

# 加载并显示第四个图像文件
img4 = mpimg.imread('img5d.png')
axes[1, 1].imshow(img4)
axes[1, 1].axis('off')  # 关闭坐标轴
axes[1, 1].text(0.5, -0.04, '(d) BIOSSES on training ratio of 10%', transform=axes[1, 1].transAxes,
                ha='center', va='center')

# 调整子图之间的间距
# plt.subplots_adjust(wspace=0.0, hspace=0.07)
plt.subplots_adjust(wspace=-0.25, hspace=0.07, top=1, bottom=0.07, left=0.00, right=1)

# 显示大图
fig.savefig('fig6.png', dpi=900)
plt.show()
