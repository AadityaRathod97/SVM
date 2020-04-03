

'''
1) Prepare support vector machines model for classifying the area under fire for foresfires data
'''



import pandas as pd 
import numpy as np 
import seaborn as sns
from timeit import default_timer as timer


forestfires = pd.read_csv("forestfires.csv")
forestfires.head()
forestfires.describe()
forestfires.columns

forestfires.drop("month",axis=1,inplace=True)
forestfires.drop("day",axis=1,inplace=True)

forestfires.size_category.value_counts()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(forestfires,test_size = 0.3)

train_X = train.iloc[:,0:28]
train_y = train.iloc[:,28]
test_X  = test.iloc[:,0:28]
test_y  = test.iloc[:,28]

# kernel = linear

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 0.980

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 0.97

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 0.71
