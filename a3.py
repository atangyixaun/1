
import pandas as pd
import numpy as np
#encoding=utf-8
churn = pd.read_csv('churn.csv')
col_names = churn.columns.tolist()  #数组转化为列表
print(col_names)

#将true值转化为0 和1
churn_result = churn_df['Churn?']
Y= np.where(churn_result=='True',1,0)  #Y为判断值
#删除排除项
drop_column = ['State','Area Code','Phone','Churn?']
X_churn = churn.drop(drop_column,axis = 1)

#X_churn['Int'l Plan'].iloc(X_churn['Int'l Plan']=='yes')=1
#X_churn['VMail Plan'].iloc(X_churn['VMail Plan']=='no')=0
churn.loc[churn["Int'l Plan"]=="yes","Int'l Plan"] =1
churn.loc[churn["Int'l Plan"]=="no","Int'l Plan"] =0
churn.loc[churn["VMail Plan"]=="yes","VMail Plan"] =1
churn.loc[churn["VMail Plan"]=="no","VMail Plan"] =0
features = X_churn.columns
X =X_churn.as_matrix().astype(np.float) #X为数据

#初始化数据,标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

#构建交叉验证函数,当作模板
from sklearn.cross_validation import KFold
def run_ev(X,Y,clf_class,**kwargs):  #数据，标签，分类器，分类器的参数
	kf = KFold(len(y),n_folds=5,shuffle = True)
	y_pred = Y.copy()
	for train,test in kf:
		X_train,X_test = X[train],X[test]
		y_train = y[train]
		clf = clf_class(**kwargs)
		clf.fit(X_train,y_train)
		y_pred[test]=clf.predict(X_test)
		
	return y_pred #返回预测值
	#print(type(y_pred))
	
from sklearn.svm import SVC  #支持向量机
from sklearn.ensemble import RandomForestClassifier as RF  #随机森林
from sklearn.neighbors import KNeighborsClassifier as KNN  #k最近邻

def accuracy(y_true,y_pred):
	return np.mean(y_true==y_pred)
	
print(accuracy(y,run_ev(X,y,SVC))) #y为真实判断值，run_ev为预测值
print(accuracy(y,run_ev(X,y,RF)))
print(accuracy(y,run_ev(X,y,KNN)))

#绘制混淆矩阵，评价模型
def cm_plot(y, yp):  
  from sklearn.metrics import confusion_matrix #导入混淆矩阵函数  
  cm = confusion_matrix(y, yp) #混淆矩阵  
  import matplotlib.pyplot as plt #导入作图库  
  plt.matshow(cm, cmap=plt.cm.Greens) #画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。  
  plt.colorbar() #颜色标签  
  for x in range(len(cm)): #数据标签  
    for y in range(len(cm)):  
      plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')  
  plt.ylabel('True label') #坐标轴标签  
  plt.xlabel('Predicted label') #坐标轴标签  
  return plt  
#绘制不同模型的混淆矩阵图,查看模型效果
cm_plot(y,run_ev(X,Y,SVC))
cm_plot(y,run_ev(X,Y,RF))
cm_plot(y,run_ev(X,Y,KNN))

def run_prob_ev(X,y,clf_class,**kwargs):
	kf = KFold(len(y),n_folds=5,shuffle = True)
	#print(kf)
	y_prob = np.zeros(len(y),2)  #numpy矩阵
	for train_index,test_index in kf:
		X_train,X_test =X[train_index],X[test_index]
		y_train = y[train_index]
		clf = clf_class(**kwargs)
		clf.fit(X_train,y_train)
		y_prob[test_index]=clf.predict_proba(X_test)#分类转换为概率值,空矩阵添加内容
		
	return y_prob
#忽略告警信息
import warnings
warnings.filterwarnings('ignore')

pred_prob = run_prob_ev(X,y,RF,n_estimators = 10)
pred_churn = pred_prob[:,1]  #预测概率值
is_churn = y == 1

counts = pd.value_counts(pred_churn) #不同概率值出现次数
true_prob=[]
for prob in counts.index:
	true_prob[prob]=np.mean(is_churn[pred_churn==prob])  #用布尔值作为索引
	true_prob = pd.Series(true_prob)
	
counts = pd.concat([counts,true_prob],axis =1).reset_index()
counts.columns = ['pred_prob','count','true_prob']
