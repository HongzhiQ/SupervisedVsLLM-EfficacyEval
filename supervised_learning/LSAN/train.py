import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def train(attention_model,train_loader,test_loader,criterion,opt,epochs = 5,GPU=True):
    F=0
    if GPU:
        attention_model.cuda()
    for i in range(epochs):
        print("Running EPOCH",i+1)
        train_loss = []
        prec_k = []
        ndcg_k = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            x, y = train[0].cuda(), train[1].cuda()
            y_pred= attention_model(x,)
            loss = criterion(y_pred, y.float())/train_loader.batch_size
            loss.backward()
            opt.step()
            labels_cpu = y.data.cpu().float()
            train_loss.append(float(loss))

        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        print("epoch %2d train end : avg_loss = %.4f" % (i+1, avg_loss))
        test_acc_k = []
        test_loss = []
        test_ndcg_k = []
        predict = np.zeros((0, 12), dtype=np.int32)
        gt = np.zeros((0, 12), dtype=np.int32)
        for batch_idx, test in enumerate(tqdm(test_loader)):
            x, y = test[0].cuda(), test[1].cuda()
            val_y= attention_model(x)
            loss = criterion(val_y, y.float()) /train_loader.batch_size
            labels_cpu = y.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            logits = np.multiply(pred_cpu.numpy() >= 0.3, 1)
            predict = np.concatenate((predict, logits))
            gt = np.concatenate((gt, labels_cpu.numpy()))
            test_loss.append(float(loss))
        recall, precision, f1 = calculate_evaluation(predict, gt, type='macro')
        print("recall", recall)
        print("precision", precision)
        print("f1", f1)
        avg_test_loss = np.mean(test_loss)
        print("epoch %2d test end : avg_loss = %.4f" % (i+1, avg_test_loss))

def calculate_evaluation(prediction,true_label,type):
    recall_list=[]
    precision_list=[]
    f1_list=[]
    for i in range(0,len(true_label)):
        recall=metrics.recall_score(true_label[i],prediction[i],average=type)
        recall_list.append(recall)
        precision=metrics.precision_score(true_label[i],prediction[i],average=type)
        precision_list.append(precision)
        f1=metrics.f1_score(true_label[i],prediction[i],average=type)
        f1_list.append(f1)
    recall_list=np.array(recall_list)
    precision_list=np.array(precision_list)
    f1_list=np.array(f1_list)
    return np.mean(recall_list),np.mean(precision_list),np.mean(f1_list)