# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:12:55 2021

@author: Owner
"""
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
data.insert(25, "User_many", User_many, True)
data.insert(26, "User_avg", User_avg, True)
data.insert(27, "EU_SALES", y, True) 

# discretization transform the raw data
kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
data["EU_SALES"] = kbins.fit_transform(data[["EU_SALES"]])

lab_enc = preprocessing.LabelEncoder()

dt_x = data.iloc[:,:27]
dt_y = pd.DataFrame(data["EU_SALES"])
dt_y = lab_enc.fit_transform(dt_y)
dt_y = pd.DataFrame(dt_y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(dt_x, dt_y, test_size=0.2 )
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
hgb_pipe = make_pipeline(
                         HistGradientBoostingClassifier())
parameters = {
  'histgradientboostingclassifier__max_iter': [1000,1200,1500],
 'histgradientboostingclassifier__learning_rate': [0.1],
 'histgradientboostingclassifier__max_depth' : [25, 50, 75],
 'histgradientboostingclassifier__l2_regularization': [1.5],
 }
#instantiate the gridsearch
hgb_grid = GridSearchCV(hgb_pipe, parameters, n_jobs=10, 
 cv=5,
 verbose=2, refit=True)
#fit on the grid 
hgb_grid.fit(dt_x, dt_y.values.ravel())

##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=dt_y.values.ravel(), y_pred=hgb_grid.predict(dt_x)):.2f}")
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test2.values.ravel(), y_pred=hgb_grid.predict(X_test2)):.2f}")
con =  confusion_matrix(y_true = y_test2, y_pred = hgb_grid.predict(X_test2))
print(con)
# =============================================================================
# test
# =============================================================================
X_test= pd.read_csv(r"C:\Users\Owner\Desktop\ML\DATA\X_test.csv")

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
data_test = pd.get_dummies(data_test,columns = ["Platform","Year_of_Release","Genre","Publisher","Rating","Critic_Count","User_Count"])

#interaction var
data_test.insert(100, "critic_many", data_test["Critic_Score"]*data_test["Critic_Count_many"], True) 
data_test.insert(101, "critic_few", data_test["Critic_Score"]*data_test["Critic_Count_few"], True) 
data_test.insert(102, "critic_avg", data_test["Critic_Score"]*data_test["Critic_Count_avg"], True) 
data_test.insert(103, "User_many", data_test["User_Score"]*data_test["User_Count_many"], True) 
data_test.insert(104, "User_few", data_test["User_Score"]*data_test["User_Count_few"], True) 
data_test.insert(105, "User_avg", data_test["User_Score"]*data_test["User_Count_avg"], True) 
x = data_test["User_Count_avg"]
column = dt_x.columns
testcol = data_test.columns
s = data_test.columns
for col in data_test.columns:
    if col not in column:
        data_test = data_test.drop(col,1)
data_test.insert(18, "Publisher_GT Interactive", 0)
data_test.insert(21, "Publisher_Russel", 0)
data_test.insert(23, "Publisher_SquareSoft", 0)
a = data_test["User_many"].copy()
data_test = data_test.drop("User_many",1)
data_test.insert(25,"User_many",a,True)


temp20= data_test["Publisher_SquareSoft"].copy()
temp21= data_test["Publisher_RedOctane"].copy()
temp22= data_test["Publisher_Square Enix"].copy()
temp23= data_test["User_many"].copy()
temp24= data_test["Publisher_Take-Two Interactive"].copy()
temp25= data_test["User_avg"].copy()

data_test = data_test.drop("Publisher_SquareSoft",1)
data_test = data_test.drop("Publisher_RedOctane",1)
data_test = data_test.drop("Publisher_Square Enix",1)
data_test = data_test.drop("Publisher_Take-Two Interactive",1)
data_test = data_test.drop("User_many",1)
data_test = data_test.drop("User_avg",1)



data_test.insert(20,"Publisher_RedOctane",temp21,True)
data_test.insert(21, "Publisher_Russel", 0)
data_test.insert(22,"Publisher_Square Enix",temp22,True)
data_test.insert(23, "Publisher_SquareSoft", 0)
data_test.insert(24,"Publisher_Take-Two Interactive",temp24,True)
data_test.insert(25,"User_many",temp23,True)
data_test.insert(26,"User_avg",temp25,True)


data_test = data_test.drop("User_many",1)

data_test = data_test.drop("Publisher_Russel",1)

b = data_test["User_avg"].copy()
data_test = data_test.drop("User_avg",1)
data_test.insert(26,"User_avg",b,True)

hgb_pipe = make_pipeline(
                         HistGradientBoostingClassifier())
parameters = {
  'histgradientboostingclassifier__max_iter': [1000,1200,1500],
 'histgradientboostingclassifier__learning_rate': [0.1],
 'histgradientboostingclassifier__max_depth' : [25, 50, 75],
 'histgradientboostingclassifier__l2_regularization': [1.5],
 }
#instantiate the gridsearch
hgb_grid = GridSearchCV(hgb_pipe, parameters, n_jobs=10, 
 cv=5,
 verbose=2, refit=True)
#fit on the grid 
hgb_grid.fit(dt_x, dt_y.values.ravel())

##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=dt_y.values.ravel(), y_pred=hgb_grid.predict(dt_x)):.2f}")
y_pred=hgb_grid.predict(data_test)