import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing
import pickle

import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

data = pd.read_csv("fetal_health.csv")

predict = "fetal_health"

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

best = 0

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
# for _ in range(1000):
#     x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
#     model = KNeighborsClassifier(n_neighbors = 3)
#     model.fit(x_train,y_train)
#     acc = model.score(x_test,y_test)
#     print(acc)

#     if acc > best:
#         best = acc
#         with open("fetal_health.pickle","wb") as f:
#             pickle.dump(model,f)

pickle_in = open("fetal_health.pickle","rb")
model = pickle.load(pickle_in)

print(best)
predicted = model.predict(x_test)

for i in range(len(x_test)):
    print("Actual: " , y_test[i] , " Predicted: " , predicted[i])

