# 資料集載入  
import pandas as pd
import numpy as np
from sklearn.model_selection import  cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pyswarm import pso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])


train = pd.read_csv('./datasets/KDDTrain+.txt', names = columns)
test = pd.read_csv('./datasets/KDDTest+.txt', names = columns)

# 資料處理
train['attack'] = train['attack'].apply(lambda x: 'attack' if x != 'normal' else x)
test['attack'] = test['attack'].apply(lambda x: 'attack' if x != 'normal' else x)


## 對training set 以及 testing set 做 one-hot encoding
# 合併訓練集和測試集以便後續處理
merged_df = pd.concat([train, test])
# 對類別型特徵進行one-hot encoding
merged_df = pd.get_dummies(merged_df, columns=['protocol_type', 'service', 'flag'], prefix="", prefix_sep="")
# 分離訓練集和測試集
train = merged_df.iloc[:len(train)]
test = merged_df.iloc[len(train):]

# 資料集取一半之後
train = train[:10000]
test = test[:4285]

X_train = train.drop(columns=['attack','level'], axis=1)
y_train = train['attack']
X_test = test.drop(columns=['attack','level'],axis=1)
y_test = test['attack']


# 特徵正規化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# pso feature selection
# 定義目標函式，該函式將用於指導 PSO 演算法搜索最優特徵子集
def objective_function(features):
    alpha = 0.99
    selected = features > 0.5
    num_selected = selected.sum()
    if num_selected == 0:
        return 0.0  # 如果沒有選擇特徵，直接返回一個大的損失值
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(SVC(), X_train_scaled[:, selected],y_train, cv=kf, n_jobs=-1).mean()  ## 嘗試使用這段cross_val_score再下面的程式碼裡
    score = 1 - accuracy
    # print(accuracy,score)
    num_features = X_train.shape[1]
    return score 


lb = [0] * X_train_scaled.shape[1]  # 下限（每個特徵的選擇狀態）
ub = [1] * X_train_scaled.shape[1]  # 上限（每個特徵的選擇狀態）


# 調參
population_size = 10
max_iters = 100
w = 0.5
c1 = 2
c2 = 2

best_features, _ = pso(objective_function, lb, ub, swarmsize=population_size, maxiter=max_iters, phip=c1, phig=c2)


# 提取最佳特徵子集
selected_features_indices = np.where(best_features > 0.5)[0]
X_train_selected = X_train_scaled[:, selected_features_indices]
X_test_selected = X_test_scaled[:, selected_features_indices]
selected_feature_names = [train.columns[i] for i in selected_features_indices if i < len(train.columns)]
print('Selected features:', ', '.join(selected_feature_names))


# 訓練分類器(SVM)
def run_SVM(X_train_selected, X_test_selected , y_train, y_test):
  start_time = time.time()
  clf = SVC(probability=True)
  clf.fit(X_train_selected, y_train)
  y_pred = clf.predict(X_test_selected)
  print('Accuracy on test set:')
  print(accuracy_score(y_test, y_pred))
  print('F1_score on test set:')
  print(f1_score(y_test, y_pred, average='weighted'))
  print('Precision on test set:')
  print(precision_score(y_test, y_pred, average='weighted'))
  print('Recall on test set:')
  print(recall_score(y_test, y_pred, average='weighted'))
  end_time = time.time()
  execution_time = end_time - start_time
  print("Execution time:", execution_time, "seconds")

print('train length:', len(train))
print("----------------------\nafter pso feature selection\n----------------------")
run_SVM(X_train.iloc[:, selected_features_indices], X_test.iloc[:, selected_features_indices], y_train, y_test)
print("-----------------------------\nall feature selection\n-----------------------------")
run_SVM(X_train, X_test, y_train, y_test)

