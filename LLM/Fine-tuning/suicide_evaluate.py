import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
true_label = np.load('ground_truth_np_hy.npy')
predict_label = np.load('predict_label_np_hy.npy')

print(true_label)
print(predict_label)
f1 = f1_score(true_label, predict_label)
predict = precision_score(true_label,predict_label)
recall = recall_score(true_label,predict_label)
accuracy = accuracy_score(true_label,predict_label)
print('f1:',f1)
print('precision',predict)
print('recall',recall)
print('accuracy',accuracy)