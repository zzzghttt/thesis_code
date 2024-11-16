from sklearn import metrics
import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(filename)s: %(levelname)s: %(message)s"
)

def calculate_auc(y_true, y_score):
    # 将真实标签和预测分数转换为 NumPy 数组
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 使用 sklearn.metrics 中的 roc_auc_score 函数计算 AUC
    auc = metrics.roc_auc_score(y_true, y_score)

    return auc

def calculate_ks(y_true, y_score):
    # 将真实标签和预测分数转换为 NumPy 数组
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 使用 sklearn.metrics 中的 roc_curve 函数计算 ROC 曲线
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)

    # 计算 KS 值
    ks = np.max(np.abs(tpr - fpr))

    return ks

def calculate_pr_auc(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_score)
    pr_auc = metrics.auc(recalls, precisions)
    
    return pr_auc

def compute_accuracy_at_recall(y_true, y_scores, target_recall):
    # 将真实标签和预测分数转换为 NumPy 数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    
    for idx in range(len(recalls) - 1, -1, -1):
        if recalls[idx] >= target_recall:
            return precisions[idx], thresholds[idx]
    
    return 0, 0

def compute_recall_at_accuracy(y_true, y_scores, target_accuracy):
    # 将真实标签和预测分数转换为 NumPy 数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    
    for idx in range(len(precisions) - 1):
        if precisions[idx] >= target_accuracy:
            return recalls[idx], thresholds[idx]
    
    return 0, 0