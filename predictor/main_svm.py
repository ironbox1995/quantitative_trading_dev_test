# TODO: 需要适配自己的数据进行进一步修改
# 代码使用了过于神奇的特征，感觉未必能行，我们后续调整
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split,cross_val_score #分割训练集和测试集；交叉验证
from sklearn.svm import SVC    #导入SVM模型
# import matplotlib.pyplot as plt

df_ori=pd.read_csv('./stock_data/000001.csv')  #载入原始数据


def make_ma(i, j, df_ori, price='close'):
    df_out=pd.DataFrame()
    for _ in range(i,j+1):
        df_out['ma_'+str(_)]=df_ori[price].rolling(_).mean()
    df_out['res']=np.where(df_ori['change']>0,1,0)
    return df_out


df_done=make_ma(3,200,df_ori)   # 生成3日到200日的移动平均线作为训练特征
df_done=df_done[1000:] # 考虑到上市之初易出异常数据，故丢弃股票上市前1000个交易日的数据，上市之初的行为我们后面有机会再分析


train_x,test_x,train_y,test_y=train_test_split(df_done.iloc[:,:-1],df_done['res'],test_size=0.3)
clf = SVC(C=2.0,kernel='linear')
clf.fit(train_x,train_y)
acc=clf.score(test_x,test_y) #根据给定数据与标签返回正确率的均值
print('SVM模型评价:',acc)