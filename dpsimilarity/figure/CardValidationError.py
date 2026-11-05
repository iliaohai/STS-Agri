import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams["font.sans-serif"] = ["Times new Roman"]

plt.figure(figsize=(6, 4), dpi=100)
Xlables = range(0, 31, 1)
Xticks = range(0, 30, 5)
SCE_NoSamp = [45, 32, 28, 39.8, 27.7, 27, 20, 20.01, 20.5, 20.7, 20.72, 18, 16.6, 13, 12, 14, 16, 15, 14.3, 14, 13.7, 12, 12.4, 14.1, 13.9, 14, 14.2, 12, 13.5, 13.7, 14]
SCE = [9.9, 7.8, 8, 8.1, 8.15, 6, 5.2, 5.1, 4.6, 4.5, 4.36, 4.4, 4.4, 4.39, 4.5, 4.4,4.1, 4.4, 4.3, 4.2, 4.3, 4.1, 4.21, 4.25, 4.2, 4.3, 4.3, 4.36, 4.4, 4.3, 4.3]
TGRU_NoSamp = [54, 22, 17, 14, 15, 12, 10, 9.5, 9.3, 9.2, 9.3, 9, 9.05, 8.9, 8.8, 9.05, 9, 8.8, 8.7, 8.65, 8.6, 8.55, 8.56, 8.4, 8.6, 8.4, 8.35, 8.65, 8.35, 8.36, 8.36]
TGRU = [13, 9.8, 7, 6, 5.6, 5.5, 5, 5, 4.6, 4.5, 4.4, 4.4, 4.4, 4.4, 4.4, 4.4, 4.4, 4.4, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3]
plt.plot(Xlables, SCE_NoSamp, c='black', label="SCE(-S)", marker='^', ms=5.5, lw=1.5)
plt.plot(Xlables, SCE, c='black', label="SCE(+FJS)", marker='o', ms=5.5, lw=1.5)
plt.plot(Xlables, TGRU_NoSamp, c='black', linestyle='--', label="TGRU(-S)", marker='v', ms=5.5, lw=1.5)
plt.plot(Xlables, TGRU, c='black', linestyle='--', label="TGRU(+FJS)", marker='*', ms=5.5, lw=1.5)
# plt.scatter(Xlables, SCE_NoSamp, c='black')
# plt.scatter(Xlables, SCE, c='black')
# plt.scatter(Xlables, TGRU_NoSamp, c='black')
# plt.scatter(Xlables, TGRU, c='black')
# plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
# plt.grid(True, linestyle='--', alpha=0.5)
temp1 = pd.DataFrame(SCE, columns=['SCE(+FJS)'])
temp2 = pd.DataFrame(SCE_NoSamp, columns=['SCE(-S)'])
temp3 = pd.DataFrame(TGRU, columns=['TGRU(+FJS)'])
temp4 = pd.DataFrame(TGRU_NoSamp, columns=['TGRU(-S)'])
total = pd.concat([temp1, temp2, temp3, temp4], axis=1)
total.to_excel('图4-10(a).xlsx')

plt.xlabel("epochs", fontdict={'size': 15})
plt.ylabel("q-error", fontdict={'size': 15})
# plt.title("Card Validation Error", fontdict={'size': 40})
plt.legend(loc = 'upper right', frameon=False)
plt.tight_layout()  # 去除边缘空白
plt.savefig(fname='图4-10(a).svg', format='svg')
plt.show()
