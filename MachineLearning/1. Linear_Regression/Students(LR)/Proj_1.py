import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('student-mat.csv', sep=';')
# print(data.head())


data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
# print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.1)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.coef_)
print(model.intercept_)

predictions = model.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])


