# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:53:11 2021

@author: shakd abrahamy

"""
# =============================================================================
#                                           Final Model 
# =============================================================================

# =============================================================================
# packges
# =============================================================================
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# =============================================================================
# import and processing data
# =============================================================================
y_train= pd.read_csv(r"C:\Users\Owner\Desktop\ML\y_train.csv")
X_train= pd.read_csv(r"C:\Users\Owner\Desktop\ML\X_train.csv")
X_test= pd.read_csv(r"C:\Users\Owner\Desktop\ML\X_train.csv")

data = X_train.copy(deep=True)
data.insert(15, "EU_SALES", y_train, True) 

#Exception screening
#data = data.drop(data[data["User_Count"] > 8000].index)
data = data.drop(data[data["Rating"] == "RP" ].index)
data = data.drop(data[data["Rating"] == "K-A"].index)
data = data.drop(data[data["Rating"] == "AO"].index)

dataFrame = pd.DataFrame(data=data, columns=['Rating','EU_SALES']);
#Feature Extraction:
data.drop('Developer',axis='columns', inplace=True)
data.drop('Reviewed',axis='columns', inplace=True)
data.drop('Name',axis='columns', inplace=True)

#distraction
for index, item in enumerate(data["Critic_Count"]):
    if item < 12 :
        data["Critic_Count"][index] = "few"
        continue
    if item >= 17 and item <= 44:
        data["Critic_Count"][index] = "avg"
        continue
    else:
        data["Critic_Count"][index] = "many"

for index, item in enumerate(data["User_Count"]):
    if item < 9 :
        data["User_Count"][index] = "few"
        continue
    if item >= 9 and item <= 150 :
        data["User_Count"][index] = "avg"
        continue
    else:
        data["User_Count"][index] = "many"  

#Feature Representation
data["Critic_Score"]=data["Critic_Score"].div(10)

#dummy var
data = pd.get_dummies(data,columns = ["Platform","Year_of_Release","Genre","Publisher","Rating","Critic_Count","User_Count"],drop_first =True)

#Wrapper Method:
#Backward Elimination
X = data.drop("EU_SALES", 1)
#interaction var
X.insert(300, "critic_many", data["Critic_Score"]*data["Critic_Count_many"], True) 
X.insert(301, "critic_few", data["Critic_Score"]*data["Critic_Count_few"], True) 
X.insert(302, "critic_avg", data["Critic_Score"]*data["Critic_Count_avg"], True) 
X.insert(303, "User_many", data["User_Score"]*data["User_Count_many"], True) 
X.insert(304, "User_few", data["User_Score"]*data["User_Count_few"], True) 
X.insert(305, "User_avg", data["User_Score"]*data["User_Count_avg"], True) 
User_many = X["User_many"]
User_avg = X["User_avg"]
y =data["EU_SALES"]
def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features


features = backward_elimination(X, y,significance_level = 0.05)
data = data.drop(columns=[col for col in data if col not in features])
data.insert(26, "User_many", User_many, True)
data.insert(27, "User_avg", User_avg, True)
data.insert(28, "EU_SALES", y, True) 

# discretization transform the raw data
kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
data["EU_SALES"] = kbins.fit_transform(data[["EU_SALES"]])

lab_enc = preprocessing.LabelEncoder()

dt_x = data.iloc[:,:28]
dt_y = pd.DataFrame(data["EU_SALES"])
dt_y = lab_enc.fit_transform(dt_y)
dt_y = pd.DataFrame(dt_y)

X_train2, X_test2, y_train2, y_test2 = train_test_split(dt_x, dt_y, test_size=0.1 , random_state = 42)
# =============================================================================
# comparing Adabost performance to DT after hyperparameter Tuning with the same data
# =============================================================================
#best config
model = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=42)
model.fit(X_train2, y_train2.values.ravel())
print(f"Accuracy: {accuracy_score(y_true=y_train2, y_pred=model.predict(X_train2)):.2f}")
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test2, y_pred=model.predict(X_test2)):.2f}")


# Create adaboost classifer object
param_grid = {
                'n_estimators': [100,200,300,400,500,600],
               'learning_rate':[0.5,0.1,1.5,2,2.5]
               }
Grid = GridSearchCV(AdaBoostClassifier(),param_grid = param_grid)

# Train Adaboost Classifer
model = Grid.fit(X_train2, y_train2.values.ravel())
best_model = model.best_estimator_
print(best_model)
train_preds = model.predict(X_train2)
##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=y_train2.values.ravel(), y_pred=train_preds):.2f}")
train_preds = model.predict(X_test2)
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test2.values.ravel(), y_pred=model.predict(X_test2)):.2f}")



# =============================================================================
# GradientBoostingClassifier
# =============================================================================
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train2, y_train2.values.ravel())
print(f"Accuracy: {accuracy_score(y_true=y_train2, y_pred=model.predict(X_train2)):.2f}")
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test2, y_pred=model.predict(X_test2)):.2f}")


param_grid = {
                'n_estimators': [200,300,400,500],
               'learning_rate':[0.5,0.1,0.01],
               'max_depth' : [8,10,12,14,16,18,20,22,24,26]
               }
Grid = GridSearchCV(GradientBoostingClassifier(),param_grid = param_grid)
model = Grid.fit(X_train2, y_train2.values.ravel())
best_model = model.best_estimator_
print(best_model)
train_preds = model.predict(X_train2)
##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=y_train2.values.ravel(), y_pred=train_preds):.2f}")
train_preds = model.predict(X_test2)
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test2.values.ravel(), y_pred=model.predict(X_test2)):.2f}")


# =============================================================================
# HistGradientBoostingClassifier
# =============================================================================
from sklearn.ensemble import HistGradientBoostingClassifier

param_grid = {
                'n_estimators': [100,200,300,400,500,600],
               'learning_rate':[0.5,0.1,1.5,2,2.5],
               'max_depth' : [8,10,12,14,16,18,20,22,24,26]
               }
Grid = GridSearchCV(HistGradientBoostingClassifier(),param_grid = param_grid)
model = Grid.fit(X_train2, y_train2.values.ravel())
best_model = model.best_estimator_
print(best_model)
train_preds = model.predict(X_train2)
##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=y_train2.values.ravel(), y_pred=train_preds):.2f}")
train_preds = model.predict(X_test2)
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test2.values.ravel(), y_pred=model.predict(X_test2)):.2f}")


# =============================================================================
# final prediction
# =============================================================================
#processing x_test
X_test= pd.read_csv(r"C:\Users\Owner\Desktop\ML\X_train.csv")

data_test = X_test.copy(deep=True)

#Feature Extraction:
data_test.drop('Developer',axis='columns', inplace=True)
data_test.drop('Reviewed',axis='columns', inplace=True)
data_test.drop('Name',axis='columns', inplace=True)

#distraction
for index, item in enumerate(data_test["Critic_Count"]):
    if item < 12 :
        data_test["Critic_Count"][index] = "few"
        continue
    if item >= 17 and item <= 44:
        data_test["Critic_Count"][index] = "avg"
        continue
    else:
        data_test["Critic_Count"][index] = "many"

for index, item in enumerate(data_test["User_Count"]):
    if item < 9 :
        data_test["User_Count"][index] = "few"
        continue
    if item >= 9 and item <= 150 :
        data_test["User_Count"][index] = "avg"
        continue
    else:
        data_test["User_Count"][index] = "many"  

#Feature Representation
data_test["Critic_Score"]=data_test["Critic_Score"].div(10)

#dummy var
data_test = pd.get_dummies(data_test,columns = ["Platform","Year_of_Release","Genre","Publisher","Rating","Critic_Count","User_Count"],drop_first =True)

#interaction var
data_test.insert(300, "critic_many", data_test["Critic_Score"]*data_test["Critic_Count_many"], True) 
data_test.insert(301, "critic_few", data_test["Critic_Score"]*data_test["Critic_Count_few"], True) 
data_test.insert(302, "critic_avg", data_test["Critic_Score"]*data_test["Critic_Count_avg"], True) 
data_test.insert(303, "User_many", data_test["User_Score"]*data_test["Critic_Count_many"], True) 
data_test.insert(304, "User_few", data_test["User_Score"]*data_test["Critic_Count_few"], True) 
data_test.insert(305, "User_avg", data_test["User_Score"]*data_test["Critic_Count_avg"], True) 
