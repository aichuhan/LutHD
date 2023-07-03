from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.metrics import explained_variance_score #导入解释回归模型的方差得分函数
from sklearn.metrics import  mean_squared_error #导入均方误差函数
from sklearn.preprocessing import StandardScaler #导入标准化处理函数
from sklearn.svm import SVR #导入支持向量回归模型
from sklearn.model_selection import train_test_split #导入数据集划分函数
from sklearn.datasets import load_boston #导入波士顿房价数据集
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame(load_boston().data) #将波士顿房价数据集的特征转换为DataFrame格式
target = load_boston().target #获取波士顿房价数据集的目标值

house_x_train, house_x_test, house_y_train, house_y_test = train_test_split(data,target, test_size=0.3) #将数据集按照7:3的比例划分为训练集和测试集

transfer = StandardScaler() #创建一个标准化处理对象
house_x_train = transfer.fit_transform(house_x_train) #对训练集数据进行标准化处理
house_x_test = transfer.fit_transform(house_x_test) #对测试集数据进行标准化处理

original_house_data_model = LinearRegression()  #创建一个线性回归模型对象
original_house_data_model.fit(house_x_train, house_y_train)  #用训练集数据对线性回归模型进行训练

house_y_predict = original_house_data_model.predict(house_x_test)  #用训练好的线性回归模型对测试集数据进行预测

rbf_svr=SVR(kernel='rbf')   #创建一个使用径向基核函数的支持向量回归模型对象
rbf_svr.fit(house_x_train, house_y_train) #用训练集数据对支持向量回归模型进行训练
rbf_svr_y_predict_1=rbf_svr.predict(house_x_test) #用训练好的支持向量回归模型对测试集数据进行预测

print(len(house_y_test))
x=range(0,152,1) #创建一个长度为106的序列，用于作为横坐标
fig=plt.figure() #创建一个图形对象
l1,=plt.plot(x,house_y_test) #绘制测试集房价的曲线，返回一个线条对象
l2,=plt.plot(x,house_y_predict) #绘制线性回归预测值的曲线，返回一个线条对象
plt.legend((l1, l2),('test','predict'),loc='lower left') #添加图例，指定图例位置为左下角
plt.title('波士顿房价预测——预测值曲线和测试集房价曲线图') #添加图形标题


x=range(0,152,1) #创建一个长度为106的序列，用于作为横坐标
fig=plt.figure() #创建一个图形对象
l1,=plt.plot(x,house_y_test) #绘制测试集房价的曲线，返回一个线条对象
l2,=plt.plot(x,rbf_svr_y_predict_1) #绘制支持向量回归预测值的曲线，返回一个线条对象
plt.legend((l1, l2),('test','predict'),loc='lower left') #添加图例，指定图例位置为左下角
plt.title('波士顿房价预测——预测值曲线和测试集房价曲线图') #添加图形标题

plt. show() #显示图形

print('解释回归模型的方差得分:',explained_variance_score(house_y_test, house_y_predict)) #打印线性回归模型的方差得分
print('均方误差:',mean_squared_error(house_y_test, house_y_predict)) #打印线性回归模型的均方误差

print('解释回归模型的方差得分:',explained_variance_score(house_y_test, rbf_svr_y_predict_1)) #打印支持向量回归模型的方差得分
print('均方误差:',mean_squared_error(house_y_test, rbf_svr_y_predict_1)) #打印支持向量回归模型的均方误差