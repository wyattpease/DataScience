import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

cc_apps = pd.read_csv('datasets/cc_approvals.data', header=None)
print(cc_apps.head())

cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

cc_apps_info = cc_apps.info()
print(cc_apps_info)
print("\n")
cc_apps.tail()

print(cc_apps.tail(17))
cc_apps = cc_apps.replace('?', np.nan)

print(cc_apps.tail(17))

# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

print(pd.isnull(cc_apps))

for col in cc_apps:
    if cc_apps[col].dtype == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

print(pd.isnull(cc_apps))

le = LabelEncoder()

for col in cc_apps.columns.values:
    if cc_apps[col].dtype=='object':
        le.fit(cc_apps[col])
        cc_apps[col]=le.transform(cc_apps[col])

print(cc_apps.head())

cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values

X,y = cc_apps[:,0:11] , cc_apps[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)
y_pred = logreg.predict(rescaledX_test)

print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_pred))

confusion_matrix(y_test, y_pred)

tol = [0.01, 0.001, 0.0001]
max_iter = [100,150,200]

param_grid = dict(tol=tol, max_iter=max_iter)
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
