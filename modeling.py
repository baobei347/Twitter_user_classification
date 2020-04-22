"""
该文件包含仅包含模型训练和保存
"""
import pymysql
import math
import warnings
import time
import pickle
import numpy as np
# import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# 取消警告
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=246)

features = np.load('feature.npy')
labels = np.load('label.npy')
print(features.shape)
print(labels.shape)

"""对数据集进行分割,split into training and test sets"""
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=40)
start = time.time()

"""开始模型训练"""
# model_ab=AdaBoostClassifier(base_estimator=svm.SVC(), n_estimators=115, learning_rate=0.7, algorithm='SAMME', random_state=0)

# model_sgd= SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, class_weight={0:0.15, 1: 0.35, 2:0.98, 3:2.57}, random_state=200,average=True, learning_rate='optimal', max_iter=5, tol=None)
# model= SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, class_weight=None, random_state=200,average=True, learning_rate='optimal', max_iter=5, tol=None)

#使用svm
# model = svm.SVC(C=1000, cache_size=2000,class_weight={0:0.5, 1: 0.8, 2:1.8, 3:3.35}, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto',max_iter=-1, probability=False,random_state=None, shrinking=True, tol=0.01, verbose=False)
# model = svm.SVC(C=100, cache_size=2000,class_weight=None , coef0=0.0,decision_function_shape='ovr', degree=3, gamma='auto',max_iter=-1, probability=False,random_state=None, shrinking=True, tol=0.01, verbose=False)

# model_dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=100, min_weight_fraction_leaf=0.0, max_features=0.99, random_state=5, max_leaf_nodes=None,
#                                min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

# model_kn=neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=200, p=1, metric='minkowski', metric_params=None, n_jobs=1)

# model_nb=BernoulliNB(alpha=1, binarize=0.0, fit_prior=True, class_prior=None)

# model=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

# model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#        beta_1=0.9, beta_2=0.999, early_stopping=False,
#        epsilon=1e-08, hidden_layer_sizes=(100, 30), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#        warm_start=False)

model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=300, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None)

# model = RandomForestClassifier(n_estimators=15, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)


rf = model.fit(X_train,y_train)#对训练集进行训练
pred = model.predict(X_test)#对测试集进行预测

print(metrics.accuracy_score(y_test, pred))#计算准确率
print(metrics.classification_report(y_test, pred))#计算各分类的准确率、召回率和F值

# 以写二进制的方式打开文件
file = open("model_three_classes_institution.pickle", "wb")
# 把模型写入到文件中
pickle.dump(model, file)
# 关闭文件
file.close()

# 5-fold cross-validation
# scoring=f1_score(y_test,pred, average='macro')
scores = cross_val_score(model, features, labels, scoring='f1_macro', cv=10)
print("cross-validation:")
print(scores)
print(scores.mean())

end = time.time()
print("所需时间为：",end - start)
