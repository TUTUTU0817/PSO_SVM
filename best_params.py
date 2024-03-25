# 資料集載入  
import pandas as pd
import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# from niapy.problems import Problem
# from niapy.task import Task
# from niapy.algorithms.basic import ParticleSwarmOptimization
# from pyswarm import pso
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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
train = train[:10000]
# 將「結果」欄的值變更為群體攻擊或正常數據
train.loc[train['attack'] != 'normal', "attack"] = 'attack'

# 繪製指定列餅圖的函數
def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize=12)
    plt.show()

# 繪製「protocol_type」和「attack」欄位的圓餅圖
pie_plot(train, ['protocol_type', 'attack'], 1, 2)