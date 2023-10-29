# -*- coding: utf-8 -*-
import pandas as pd

true_df = pd.read_csv("../../data/cognitive distortion/cognitive_distortion_val_LLM.csv", encoding="GBK")
pred_df = pd.read_csv("your predict file path")

true_df = true_df.sort_values(by="id")
pred_df = pred_df.sort_values(by="id")

cols_to_compare = ["非此即彼", "以偏概全", "心理过滤", "否定正面思考", "读心术", "先知错误", "放大", "情绪化推理", "应该式", "乱贴标签", "罪责归己", "罪责归他"]

true_df = true_df.sort_values(by="id")
predict_df = pred_df.sort_values(by="id")

labels = ["非此即彼", "以偏概全", "心理过滤", "否定正面思考", "读心术", "先知错误", "放大", "情绪化推理", "应该式", "乱贴标签", "罪责归己", "罪责归他"]
true_labels = true_df[labels].values
print(true_labels)
predict_labels = predict_df[labels].values

TP = ((true_labels == 1) & (predict_labels == 1)).sum()
FP = ((true_labels == 0) & (predict_labels == 1)).sum()
FN = ((true_labels == 1) & (predict_labels == 0)).sum()
TN = ((true_labels == 0) & (predict_labels == 0)).sum()

print(f"TP (True Positive) 数量: {TP}")
print(f"FP (False Positive) 数量: {FP}")
print(f"FN (False Negative) 数量: {FN}")
print(f"TN (True Negative) 数量: {TN}")

precision = TP / (TP + FP)
recall = TP / (TP + FN)

micro_f1 = 2 * precision * recall / (precision + recall)

print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"Micro F1 值: {micro_f1:.4f}")
