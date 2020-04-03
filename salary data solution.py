'''
2) Prepare a classification model using SVM for salary data 
'''


import pandas as pd 
import numpy as np 
import seaborn as sns
from timeit import default_timer as timer


Salary_data_test = pd.read_csv("SalaryData_Test.csv")
Salary_data_train = pd.read_csv("SalaryData_Train.csv")

col_list = Salary_data_test.columns

from sklearn import preprocessing
for i in col_list:
    number = preprocessing.LabelEncoder()
    Salary_data_test[i] = number.fit_transform(Salary_data_test[i])
    Salary_data_train[i] = number.fit_transform(Salary_data_train[i])



from sklearn.svm import SVC

train_X = Salary_data_train.iloc[:,0:13]
train_y = Salary_data_train.iloc[:,13]
test_X  = Salary_data_test.iloc[:,0:13]
test_y  = Salary_data_test.iloc[:,13]

# kernel = linear

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 0.80

# Kernel = poly
model_poly = SVC(kernel = "sigmoid",C=1, gamma=1)
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 0.75

# kernel = rbf
model_rbf = SVC(kernel = "rbf",C=1, gamma=1)
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 0.76
