import tkinter as tk
import tkinter.ttk as ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib as plt
import numpy as np
import pandas as pd
import math
import glob
import csv
import os

#MLについて
#処理済みのcsvからデータを読み込み->機械学習を実行
#入力
#path:csvのあるディレクトリ
#"./*.csv"
#"data/*.csv"
#C://Users//username//Desktop//work//data/*.csv
#など
#
#id:どの機械学習モデルを使用するか
#
#n:ディレクトリの何番目のcsvを使うか
#
#limit：データを何個ずつ集めるか
#
#answer:ターゲットの名前
#target
#
#*args:特徴量を任意の数入力(現在3個)
#"feature0","feature1","feature2"
#
def ML(path,id,n,limit,answer,*args):
    #データの読み込み
    csv_path = glob.glob(path)
    csv_read = pd.read_csv(csv_path[n])
    count_target0 = 0
    count_target1 = 0
    target = np.empty((0,1),int)
    feature = np.empty((0,len(args)),int)
    feature_name = np.array([*args])

    for i in range(len(csv_read)):
        tmp_target = pd.DataFrame(csv_read).iat[i,csv_read.columns.get_loc(answer)]
        tmp_feature = np.empty((0,len(args)),int)
        if(limit < count_target0 and limit < count_target1):break
        elif(tmp_target == 0 and limit < count_target0):pass
        elif(tmp_target == 1 and limit < count_target1):pass
        elif(tmp_target == 0 and count_target0 < limit):
            count_target0 += 1
            target = np.append(target,tmp_target)
            tmp_feature = [pd.DataFrame(csv_read).iat[i, csv_read.columns.get_loc(args[j])] for j in range(len(args))]
            feature = np.append(feature,[tmp_feature],axis=0)
        elif(tmp_target == 1 and count_target1 < limit):
            count_target1 += 1
            target = np.append(target,tmp_target)
            tmp_feature = [pd.DataFrame(csv_read).iat[i, csv_read.columns.get_loc(args[j])] for j in range(len(args))]
            feature = np.append(feature,np.array([tmp_feature]),axis=0)
        else:
            pass
        
    #print(target)
    #print(feature)
    #print(feature_name)

    #データの仕分け
    X = feature
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=0)

    #モデルの選択
    if id == "LogisticRegression":
        MODEL = LogisticRegression()
        ML_name = "LogisticRegression"
        MODEL.fit(X_train, y_train)
    elif id == "SVM":
        MODEL = svm.SVC()
        ML_name = "SVM"
        MODEL.fit(X_train, y_train)
    else:
        MODEL = LogisticRegression()
        ML_name = "LogisticRegression"
        MODEL.fit(X_train, y_train)

    #評価
    #少数第四位以下を切り捨て
    y_prediction = MODEL.predict(X_test)
    TN,FP,FN,TP = confusion_matrix(y_test,y_prediction).flatten()
    accuracy_score_ = math.floor(1000*((TP+TN)/(TP+TN+FP+FN)))/1000
    precision_score_ = math.floor(1000*((TP)/(TP+FP)))/1000
    recall_score_ = math.floor(1000*((TP)/(TP+FN)))/1000
    f1_score_ = math.floor(1000*((2*TP)/(2*TP+FP+FN)))/1000

    print("Feature " + str(feature_name))
    print("Accuracy  rate : " + str(accuracy_score_))
    print("Precision rate : " + str(precision_score_))
    print("Recall    rate : " + str(recall_score_))
    print("F1-score  rate : " + str(f1_score_))
    print("Number  of  0  : " + str(np.sum(target==0)))
    print("Number  of  1  : " + str(np.sum(target==1)))
    print("\n")
    
    #csvに保存
    csv_data = np.empty((0,len(args)+10),str)
    csv_data = np.array(np.array([str(ML_name)]))
    csv_data = np.append(csv_data,np.array([str(accuracy_score_)]),axis=0)
    csv_data = np.append(csv_data,np.array([str(precision_score_)]),axis=0)
    csv_data = np.append(csv_data,np.array([str(recall_score_)]),axis=0)
    csv_data = np.append(csv_data,np.array([str(f1_score_)]),axis=0)

    csv_data = np.append(csv_data,np.array([str(np.sum(target==0))]),axis=0)
    csv_data = np.append(csv_data,np.array([str(np.sum(target==1))]),axis=0)
    csv_tmp  = [str(feature_name[i]) for i in range(len(args))]
    csv_data = np.append(csv_data,np.array(csv_tmp),axis=0)

    csv_flag =  not os.path.exists("output.csv")
    with open("output.csv","a",newline = "") as f:
        if csv_flag:
            csv.writer(f).writerow(["Model","Accuracy","Precision","Recall","F1-score","Num of 0","Num of 1","F0","F1","F2"])
        csv.writer(f).writerow(csv_data)

def ML_RUN():
    #プルダウンの入力内容を受け取り,MLを実行
    #入力例ML("data/*.csv","LogisticRegression",0,10000,"target","feature0","feature2","feature4")
    ML(pulldown0.get(),pulldown1.get(),int(pulldown2.get()),int(pulldown3.get()),pulldown4.get(),pulldown5.get(),pulldown6.get(),pulldown7.get())

    #結果の出力はML
    #記録自体はcsvでも出力されている


#ウィンドウ設定
root = tk.Tk()
root.title("機械学習実行")
#root.geometry("400x400")


#関数MLへの入力をプルダウンで受付
tk.Label(root,text="directory",font=("Helvetica",10)).grid(row=0,column=0)
pulldown0 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["data/*.csv","./*.csv",""])
pulldown0.set("data/*.csv")
pulldown0.grid(row=0,column=1)

tk.Label(root,text="model",font=("Helvetica",10)).grid(row=1,column=0)
pulldown1 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["LogisticRegression","SVM",""])
pulldown1.set("LogisticRegression")
pulldown1.grid(row=1,column=1)

tk.Label(root,text="nth",font=("Helvetica",10)).grid(row=2,column=0)
pulldown2 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["0"])
pulldown2.set("0")
pulldown2.grid(row=2,column=1)

tk.Label(root,text="limit",font=("Helvetica",10)).grid(row=3,column=0)
pulldown3 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["1000","5000","10000"])
pulldown3.set("1000")
pulldown3.grid(row=3,column=1)

tk.Label(root,text="target",font=("Helvetica",10)).grid(row=4,column=0)
pulldown4 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["target"])
pulldown4.set("target")
pulldown4.grid(row=4,column=1)

tk.Label(root,text="feature0",font=("Helvetica",10)).grid(row=5,column=0)
pulldown5 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["feature0","feature1","feature2","feature3","feature4","feature5"])
pulldown5.set("SPORT")
pulldown5.grid(row=5,column=1)

tk.Label(root,text="feature1",font=("Helvetica",10)).grid(row=6,column=0)
pulldown6 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["feature0","feature1","feature2","feature3","feature4","feature5"])
pulldown6.set("DPORT")
pulldown6.grid(row=6,column=1)

tk.Label(root,text="feature2",font=("Helvetica",10)).grid(row=7,column=0)
pulldown7 = ttk.Combobox(root, textvariable=tk.StringVar(),values=["feature0","feature1","feature2","feature3","feature4","feature5"])
pulldown7.set("Bytes")
pulldown7.grid(row=7,column=1)

run_button = tk.Button(root, text="RUN!",command=ML_RUN)
run_button.grid(row=8)

root.mainloop()
