# -*- coding: utf-8 -*-


# =============================================================================
# SVM
# =============================================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

y_train_handled = pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\y_train_handled.csv")
X_train_handled= pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\X_train_handled.csv")

X_train, X_test, y_train, y_test = train_test_split(X_train_handled, y_train_handled, test_size=0.2, random_state=123)

model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)



print("----------------\n")
print(f"Train Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.4f}")
print(f"Test Accuracy: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test)):.4f}")
print("----------------\n")



param_grid = {'C': np.arange(0, 2, 0.1) ,
              'kernel': ['linear', 'poly','rbf','sigmoid']
              }


param_grid = {'C': np.arange(5, 10, 0.1),
              'kernel': ['linear'],
              }

grid_search = GridSearchCV(estimator=SVC(random_state=42),
                            param_grid=param_grid,
                            refit=True,
                            cv=10)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print('\n-------------')
print('Best Paramters:')
print(grid_search.best_params_)
print('------------- \n')

train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)
print("Train accuracy: ", accuracy_score(y_true=y_train, y_pred=train_preds))
print("Test accuracy: ", accuracy_score(y_true=y_test, y_pred=test_preds))

model = SVC(kernel='linear', C=9.89)
model.fit(X_train, y_train)

print('w = ',best_model.coef_)
print('b = ',best_model.intercept_)
print('Coefficients of the support vector in the decision function = ', np.abs(best_model.dual_coef_))

from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import datetime

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
    

features_names = ['NA_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'User_Score', 'Platform_GC', 'Platform_PC', 'Platform_PS', 'Platform_PS3', 'Platform_PS4', 'Platform_Wii', 'Year_of_Release_1997', 'Year_of_Release_2002', 'Year_of_Release_2003', 'Year_of_Release_2013', 'Year_of_Release_2014', 'Genre_Fighting', 'Genre_Racing', 'Genre_Role-Playing', 'Publisher_GT Interactive', 'Publisher_Microsoft Game Studios', 'Publisher_RedOctane', 'Publisher_Russel', 'Publisher_Square Enix', 'Publisher_SquareSoft', 'Publisher_Take-Two Interactive', 'User_many', 'User_avg']
model = SVC(kernel='linear',C=9.89)
model.fit(X_train, y_train)

for i in model.coef_:
    f_importances(i, features_names)

# =============================================================================
# compre SVM to other models
# =============================================================================
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

prdict_y = pd.DataFrame(model.predict(X_test), columns = ['prdict_y'])
print(confusion_matrix(y_true = y_test, y_pred = prdict_y))
TP = confusion_matrix(y_true = y_test, y_pred = prdict_y)[0,0]
FP = confusion_matrix(y_true = y_test, y_pred = prdict_y)[0,1]
FN = confusion_matrix(y_true = y_test, y_pred = prdict_y)[1,0]
TN = confusion_matrix(y_true = y_test, y_pred = prdict_y)[1,1]
#True Positive Rate (TPR) or Hit Rate or Recall or Sensitivity = TP / (TP + FN)
TPR = TP / (TP + FN)
#False Positive Rate(FPR) or False Alarm Rate = 1 - Specificity = 1 - (TN / (TN + FP))
FPR = 1 - (TN / (TN + FP))
Precision = TP / (TP + FP)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, prdict_y)
print(TPR,FPR)
auc = roc_auc_score(y_test, prdict_y)
print('AUC: %.3f' % auc)
