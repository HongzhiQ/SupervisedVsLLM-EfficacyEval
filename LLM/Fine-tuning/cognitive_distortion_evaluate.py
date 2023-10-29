import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

def calculate_evaluation(prediction,true_label,type = 'binary'):
    recall_list=[]
    precision_list=[]
    f1_list=[]
    for i in range(0,len(true_label)):
        recall=recall_score(true_label[i],prediction[i],average=type)
        recall_list.append(recall)
        precision=precision_score(true_label[i],prediction[i],average=type)
        precision_list.append(precision)
        f1=f1_score(true_label[i],prediction[i],average=type)
        f1_list.append(f1)
    recall_list=np.array(recall_list)
    precision_list=np.array(precision_list)
    f1_list=np.array(f1_list)
    return np.mean(recall_list),np.mean(precision_list),np.mean(f1_list)

true_label = [[0,1,0,0],[1,0,1,1],[1,0,1,0]]
predict_label = [[0,1,0,1],[1,0,0,1],[1,0,1,1]]
true_label = np.load('ground_truth_np_2.npy')
predict_label = np.load('predict_label_np_2.npy')


print(true_label)
print(predict_label)
accuracy = accuracy_score(true_label,predict_label)
macro_f1 = f1_score(true_label,predict_label,average='macro')
micro_f1 = f1_score(true_label,predict_label,average='micro')
samples_f1 = f1_score(true_label,predict_label,average='samples')
_,_,f1 = calculate_evaluation(true_label,predict_label)
precision = precision_score(true_label,predict_label,average='micro')
recall = recall_score(true_label,predict_label,average='micro')

print('macro_f1:',macro_f1)
print('micro_f1',micro_f1)
print('samples_f1',samples_f1)
print('f1',f1)

print('recall',recall)
print('precision',precision)

