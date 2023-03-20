#阿里云天池数据挖掘竞赛，工业蒸汽数据回归
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

#加载数据集
def load_data():
    Train_X=pd.read_csv('data/蒸汽/zhengqi_train.txt',sep='\t')
    Test_X=pd.read_csv('data/蒸汽/zhengqitest.txt',sep='\t')
    Train_Y=Train_X.iloc[:,-1]
    Train_Y=pd.DataFrame(Train_Y,columns=['target'])
    del Train_X['target']
    return Train_X,Train_Y,Test_X

if __name__=='__main__':
    train_x,train_y,test_x=load_data()
    col=train_x.columns.tolist()
    featurenums=len(col)

    #删除异常数据,根据3σ法则删除数据
    for i in range(featurenums):
        Data=train_x.iloc[:,i]
        u=Data.mean()
        theta=Data.std()
        bidx=((Data<u-3*theta)|(Data>u+3*theta))
        idx=np.arange(train_x.shape[0])[bidx]
        if len(idx)!=0:
            train_x.drop(index=idx,inplace=True)
            train_y.drop(index=idx,inplace=True)
            x_array=np.array(train_x)
            y_array=np.array(train_y)
            train_x=pd.DataFrame(x_array,index=np.arange(len(x_array)),columns=col)
            train_y = pd.DataFrame(y_array, index=np.arange(len(y_array)), columns=['target'])

    #删除分布不同的特征
    wrong_feature=['V5','V9','V11','V17','V22','V28']
    for item in wrong_feature:
        train_x.drop(labels=item,axis=1,inplace=True)
        test_x.drop(labels=item,axis=1,inplace=True)

    #切分训练集与测试集
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    train_data,test_data,train_label,test_label=train_test_split(train_x,train_y,test_size=0.1)

    #PCA降维
    from sklearn.decomposition import PCA
    PCA_MOD=PCA(n_components=0.99,svd_solver='full')
    lowd_data=PCA_MOD.fit_transform(train_data)
    print(np.shape(lowd_data))

    #线性模型回归
    Lin=LinearRegression()
    Lin.fit(lowd_data,train_label)

    #Lasso回归
    lasso=Lasso(alpha=0.0025)
    lasso.fit(lowd_data,train_label)

    #KNN模型回归
    knn=KNeighborsRegressor(n_neighbors=12)
    knn.fit(lowd_data,train_label)

    #xgboost模型回归
    XGB=xgb.XGBRegressor(max_depth=2,n_estimators=500)
    XGB.fit(lowd_data,train_label)

    #在测试集上进行预测
    from sklearn.metrics import mean_squared_error
    lowd_test=PCA_MOD.transform(test_data)
    y_pred_xgb=XGB.predict(lowd_test)
    y_pred_lin=Lin.predict(lowd_test)
    y_pred_las=lasso.predict(lowd_test)
    y_pred_knn=knn.predict(lowd_test)

    #评估模型的泛化性能
    print('XGboost的MSE:',mean_squared_error(test_label,y_pred_xgb))
    print('线性回归MSE:',mean_squared_error(test_label,y_pred_lin))
    print('lasso回归MSE:', mean_squared_error(test_label, y_pred_las))
    print('Knn回归MSE:', mean_squared_error(test_label, y_pred_knn))

    #选择效果最好的模型对题目数据进行预测
    lowd_x=PCA_MOD.transform(test_x)
    y_pred_test=Lin.predict(lowd_x)
    np.savetxt('data/蒸汽/result.txt',y_pred_test,fmt='%f')

