from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd

# AUC的95%置信区间
def AUC_CI(auc, label, alpha = 0.05):
	label = np.array(label)#防止label不是array类型
	n1, n2 = np.sum(label == 1), np.sum(label == 0)
	q1 = auc / (2-auc)
	q2 = (2 * auc ** 2) / (1 + auc)
	se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2))
	confidence_level = 1 - alpha
	z_lower, z_upper = norm.interval(confidence_level)
	lowerb, upperb = auc + z_lower * se, auc + z_upper * se
	return (lowerb, upperb)



# 曲线一
roc_bertlog = pd.read_csv('../../output/dpsimilarity/run_mrpc_result_training_ratio0.05.csv', header=0)  # 训练集
roc_bertlog_y_true = roc_bertlog['y_true'].values  # 正确值
roc_bertlog_y_pred = roc_bertlog['y_pred'].values  # 预测值
FPR_bertlog, TPR_bertlog, _ = roc_curve(roc_bertlog_y_true, roc_bertlog_y_pred, pos_label = 1)
AUC_bertlog = roc_auc_score(roc_bertlog_y_true, roc_bertlog_y_pred)
CI_bertlog = AUC_CI(roc_bertlog_y_true, roc_bertlog_y_pred, 0.05)

# 曲线二
DecisionTree = pd.read_csv('../../output/distilbert/run_mrpc_result_training_ratio0.05.csv', header=0)  # 训练集
DecisionTree_y_true = DecisionTree['y_true'].values  # 正确值
DecisionTree_y_pred = DecisionTree['y_pred'].values  # 预测值
FPR_DecisionTree, TPR_DecisionTree, _ = roc_curve(DecisionTree_y_true, DecisionTree_y_pred, pos_label = 1)
AUC_DecisionTree = roc_auc_score(DecisionTree_y_true, DecisionTree_y_pred)
CI_DecisionTree = AUC_CI(DecisionTree_y_true, DecisionTree_y_pred, 0.05)

# 曲线三
deeplog = pd.read_csv('../../output/spanbert/run_mrpc_result_training_ratio0.05.csv', header=0)  # 训练集
deeplog_y_true = deeplog['y_true'].values  # 正确值
deeplog_y_pred = deeplog['y_pred'].values  # 预测值
FPR_deeplog, TPR_deeplog, _ = roc_curve(deeplog_y_true, deeplog_y_pred, pos_label = 1)
AUC_deeplog = roc_auc_score(deeplog_y_true, deeplog_y_pred)
CI_deeplog = AUC_CI(deeplog_y_true, deeplog_y_pred, 0.05)

# 曲线四
result_InvariantsMiner = pd.read_csv('../../output/t5/run_mrpc_result_training_ratio0.05.csv', header=0)  # 训练集
result_InvariantsMiner_y_true = result_InvariantsMiner['y_true'].values  # 正确值
result_InvariantsMiner_y_pred = result_InvariantsMiner['y_pred'].values  # 预测值
FPR_result_InvariantsMiner, TPR_result_InvariantsMiner, _ = roc_curve(result_InvariantsMiner_y_true, result_InvariantsMiner_y_pred, pos_label = 1)
AUC_result_InvariantsMiner = roc_auc_score(result_InvariantsMiner_y_true, result_InvariantsMiner_y_pred)
CI_result_InvariantsMiner = AUC_CI(result_InvariantsMiner_y_true, result_InvariantsMiner_y_pred, 0.05)

# 曲线五
result_LR = pd.read_csv('../../output/bert/run_mrpc_result_training_ratio0.05.csv', header=0)  # 训练集
result_LR_y_true = result_LR['y_true'].values  # 正确值
result_LR_y_pred = result_LR['y_pred'].values  # 预测值
FPR_result_LR, TPR_result_LR, _ = roc_curve(result_LR_y_true, result_LR_y_pred, pos_label = 1)
AUC_result_LR = roc_auc_score(result_LR_y_true, result_LR_y_pred)
CI_result_LR = AUC_CI(result_LR_y_true, result_LR_y_pred, 0.05)

# 曲线六
# result_SVM = pd.read_csv('../../../Dataset/HDFS/ROC_Output/result_SVM.csv', header=0)  # 训练集
# result_SVM_y_true = result_SVM['y_true'].values  # 正确值
# result_SVM_y_pred = result_SVM['y_pred'].values  # 预测值
# FPR_result_SVM, TPR_result_SVM, _ = roc_curve(result_SVM_y_true, result_SVM_y_pred, pos_label = 1)
# AUC_result_SVM = roc_auc_score(result_SVM_y_true, result_SVM_y_pred)
# CI_result_SVM = AUC_CI(result_SVM_y_true, result_SVM_y_pred, 0.05)


# 开始画线
fig = plt.figure(figsize=(3.2, 3))
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.94, top=0.94)
f_size = 7
ax = plt.subplot(1, 1, 1)
ax.tick_params(width=0.5, length=2)
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.plot(FPR_bertlog, TPR_bertlog, lw=0.6, label="DPSimilarity AUC = %0.3f" % AUC_bertlog)
plt.plot(FPR_deeplog, TPR_deeplog, lw=0.6, label="DistilBERT AUC = %0.3f" % AUC_deeplog)
plt.plot(FPR_result_LR, TPR_result_LR, lw=0.6, label="SpanBERT AUC = %0.3f" % AUC_result_LR)
plt.plot(FPR_result_InvariantsMiner, TPR_result_InvariantsMiner, lw=0.6, label="T5 AUC = %0.3f" % AUC_result_InvariantsMiner)
plt.plot(FPR_result_LR, TPR_result_LR, lw=0.6, label="BERT AUC = %0.3f" % AUC_result_LR)

#plt.rcParams.update({'font.size': f_size})
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=f_size)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=f_size)
plt.xlabel('False Positive Rate', horizontalalignment="center", verticalalignment="center", fontsize=f_size)
plt.ylabel('True Positive Rate', horizontalalignment="center", verticalalignment="center", fontsize=f_size)

#ax = plot_AUC(FPR_bertlog, TPR_bertlog, AUC_bertlog, CI_bertlog, label = 'BERT-Log')
# ax = plot_AUC(ax, FPR_DecisionTree, TPR_DecisionTree, AUC_DecisionTree, CI_DecisionTree, label = 'DecisionTree')
#ax = plot_AUC(FPR_deeplog, TPR_deeplog, AUC_deeplog, CI_deeplog, label = 'Deeplog')
# ax = plot_AUC(ax, FPR_result_InvariantsMiner, TPR_result_InvariantsMiner, AUC_result_InvariantsMiner, CI_result_InvariantsMiner, label = 'InvariantsMiner')
#ax = plot_AUC(FPR_result_LR, TPR_result_LR, AUC_result_LR, CI_result_LR, label = 'LR')
#ax = plot_AUC(FPR_result_SVM, TPR_result_SVM, AUC_result_SVM, CI_result_SVM, label = 'SVM')

#ax.plot((0, 1), (0, 1), ':', color = 'grey')
#plt.title("ROC based on HDFS", fontsize=8)
plt.legend(fontsize=f_size)
# plt.savefig("d:/software/Figure9.jpg", dpi=1200)
plt.show()
plt.close(fig)
