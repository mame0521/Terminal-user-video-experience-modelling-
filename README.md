# End user video experience modelling


### Question A: Network side estimation end-user video experience modeling

With the upgrading of wireless broadband network and the popularity of intelligent terminals, more and more users choose to watch network videos on mobile intelligent terminals with application client APP, which is a TCP-based video transmission and playback. The two key indicators for network video to affect user experience are the initial buffer waiting time and the lag buffer time in the process of video playback. We can use the initial buffer delay and the proportion of lag time (lag time ratio = lag time/video playback time) to quantitatively evaluate user experience. Some studies have shown that the main factors that affect the initial buffer delay and the proportion of cache time are the initial buffer peak rate, the average download rate of playback phase, the end to end loop back time (E2E RTT), and video parameters. However, the relationship between these factors and the initial delay and the proportion of catton time is not clear.
According to the experimental data provided in the attachment, please try to establish the functional relationship between the user experience evaluation variables (initial buffer delay, proportion of cache time) and the network side variables (initial buffer peak rate, average download rate during playback phase, E2E RTT).



![image-20230309034836914](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/image-20230309034836914.png)




```python
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

###########1.数据生成部分##########
def f(x1, x2):
    y = 0.5 *np.power(x1,-2) + 0.5 * np.cos(x2) + 3 + 0.1 * x1
    return y

df = pd.read_csv('speed_vedio.csv')
df = df[[1,2,3,4]]
x_train= df.iloc[:82000,0:3]
y_train =df.iloc[:82000,3]#数据前两列是x1,x2 第三列是y,这里的y 有随机噪声
x_test ,y_test = df.iloc[82000:,:3], df.iloc[82000:,3] # 同上,不过这里的y 没有噪声
from sklearn.metrics import r2_score

def try_different_method(model):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()
    
    from sklearn import tree
    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
    from sklearn import linear_model
    model_LinearRegression = linear_model.LinearRegression()
    from sklearn import svm
    model_SVR = svm.SVR()
    from sklearn import neighbors
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    from sklearn import ensemble
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20 个决策树
    from sklearn import ensemble
    model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50 个决策树
    from sklearn import ensemble
    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100 个决策树
    from sklearn.ensemble import BaggingRegressor
    model_BaggingRegressor = BaggingRegressor()
    from sklearn.tree import ExtraTreeRegressor
    model_ExtraTreeRegressor = ExtraTreeRegressor()
    
try_different_method(model_LinearRegression)

import pandas as pd
df = pd.read_csv('speed_vedio.csv')
print(df.head())
df = df[[1,2,3,4,5]]
print(df.head())
df.to_csv('speed_vedio.csv')
df=df[df['kadun_percent']>0]
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
k = 5 #number ofvariables for heatmap
cols = corrmat.nlargest(k, 'start_time_wait')['start_time_wait'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
yticklabels=cols.values, xticklabels=cols.values)
plt.show()
data1 = df[[0,1,2,3]]
X = data1[['start_speed','E2E RTT','avr_bo_speed']]
y = data1['start_time_wait']
y = data1.start_time_wait

import numpy as np
from sklearn.cross_validation import train_test_split #这里是引用了交叉验证
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.04)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
feature_cols = ['start_speed','E2E RTT','avr_bo_speed']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model=linreg.fit(X_train, y_train)
print (model)
print (linreg.intercept_)
print (linreg.coef_ )
print(feature_cols, linreg.coef_)
y_pred = linreg.predict(X_test)
print (y_pred)

from sklearn import metrics
rms=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print (np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import seaborn as sns
#第一个变了幂函数回归拟合
df = pd.read_csv('speed_vedio.csv')
df = df[[1,2,3,4,5]]
def func(x, a, b, c):
    return a * np.power(b,x) + c
ydata = df['start_time_wait']
xdata = df['start_speed']
popt, pcov = curve_fit(func, xdata, ydata)
#popt 数组中，三个值分别是待求参数a,b,c
y2 = [func(i, popt[0],popt[1],popt[2]) for i in xdata]
plt.plot(xdata,y2,'r--')
print (popt)
plt.show()
xdata = np.array(xdata )
y_pre = 1.52100472e+04 * np.power(9.99803646e-01,xdata) + 1.06102126e+03
print(y_pre)
print(r2_score(ydata,y_pre))
xdata = np.linspace(0,160000,10000)
sns.pairplot(df, x_vars=['start_speed'], y_vars='start_time_wait',size=10, aspect=0.8)
plt.plot(xdata,1.52100472e+04 * np.power(9.99803646e-01,xdata) + 1.06102126e+03,'r--')
plt.show()
sns.pairplot(data1.dropna())
plt.show()
sns.pairplot(data, x_vars=['start_speed','E2E RTT','avr_bo_speed'], y_vars='start_time_wait',
size=10, aspect=0.8)
plt.show()
#data=(data-data.min())/(data.max()-data.min())#normalization
sns.pairplot(data, x_vars=['start_speed','E2E RTT','avr_bo_speed'], y_vars='start_time_wait',
size=10, aspect=0.8)
plt.show()


```

