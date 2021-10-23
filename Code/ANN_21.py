# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:34:29 2020

@author: Owner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer 
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#import data

y_train= pd.read_csv(r"C:\Users\Owner\Desktop\ML\y_train.csv")
X_train= pd.read_csv(r"C:\Users\Owner\Desktop\ML\X_train.csv")
#######'X_train'################################################################

data = X_train.copy(deep=True)
data.insert(15, "EU_SALES", y_train, True) 

#Exception screening
data = data.drop(data[data["User_Count"] > 8000].index)
data = data.drop(data[data["Rating"] == "RP" ].index)
data = data.drop(data[data["Rating"] == "K-A"].index)
data = data.drop(data[data["Rating"] == "AO"].index)

dataFrame = pd.DataFrame(data=data, columns=['Rating','EU_SALES']);
dataFrame.plot.scatter(x='Rating', y='EU_SALES', title= "Scatter plot between Rating to EU_SALES ");

plt.show(block=True);
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

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["EU_SALES"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
relevant_features

print(data[["NA_Sales","JP_Sales","Other_Sales"]].corr())


#dummy var
data = pd.get_dummies(data,columns = ["Platform","Year_of_Release","Genre","Publisher","Rating","Critic_Count","User_Count"],drop_first =True)

#Wrapper Method:
#Backward Elimination
X = data.drop("EU_SALES", 1)
#interaction var
X.insert(300, "critic_many", data["Critic_Score"]*data["Critic_Count_many"], True) 
X.insert(301, "critic_few", data["Critic_Score"]*data["Critic_Count_few"], True) 
X.insert(302, "critic_avg", data["Critic_Score"]*data["Critic_Count_avg"], True) 
X.insert(303, "User_many", data["User_Score"]*data["Critic_Count_many"], True) 
X.insert(304, "User_few", data["User_Score"]*data["Critic_Count_few"], True) 
X.insert(305, "User_avg", data["User_Score"]*data["Critic_Count_avg"], True) 
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

# =============================================================================
# ####################### PART 2  ################
# =============================================================================

# discretization transform the raw data
kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
data["EU_SALES"] = kbins.fit_transform(data[["EU_SALES"]])

lab_enc = preprocessing.LabelEncoder()

dt_x = data.iloc[:,:28]
dt_y = pd.DataFrame(data["EU_SALES"])
dt_y = lab_enc.fit_transform(dt_y)
dt_y = pd.DataFrame(dt_y)

X_train, X_test, y_train, y_test = train_test_split(dt_x, dt_y, test_size=0.2 , random_state = 42)

#====== Standardizing ======

def standardize_set():

     # ----- Exporting Column Mean & STD-----
    
    columns_to_norm = ['NA_Sales', 'JP_Sales', 'Other_Sales','Critic_Score','User_Score','User_many','User_avg']
    
    train_column_mean_std = pd.DataFrame()
    test_column_mean_std = pd.DataFrame()
    for column in columns_to_norm:
        train_column_mean_std  = train_column_mean_std.append({'column': column,
                                                    'Mean': X_train[column].mean(),
                                                    'STD': X_train[column].std()}, ignore_index=True)
        
        test_column_mean_std = test_column_mean_std.append({'column': column,
                                                    'Mean': X_test[column].mean(),
                                                    'STD': X_test[column].std()}, ignore_index=True)
        
    train_column_mean_std = train_column_mean_std[['column', 'Mean', 'STD']]
    test_column_mean_std = test_column_mean_std[['column', 'Mean', 'STD']]
    
    print(train_column_mean_std)
    
     # ----- Standardizing ----- 
    scaler = StandardScaler()  
    X_train_standard = X_train
    X_test_standard = X_test

    X_train_standard[columns_to_norm] = scaler.fit_transform(X_train[columns_to_norm])
    X_test_standard[columns_to_norm] = scaler.fit_transform(X_test[columns_to_norm])

    
standardize_set()

# =============================================================================
#  ====== 3.1 Building a default ANN ======
# =============================================================================

def default_ANN(plot=False):
    
     # ----- Training model with default values -----

     model = MLPClassifier(random_state=42,
                           hidden_layer_sizes =(28),
                           activation='relu',
                           verbose=False,
                           early_stopping=False,
                           max_iter=200,
                           learning_rate_init=0.001)
     model.fit(X_train,y_train)

     print("----------------\n")
     print(f"Train Accuracy: {accuracy_score(y_true=y_train.values.ravel(), y_pred=model.predict(X_train)):.4f}")
     print(f"Test Accuracy: {accuracy_score(y_true=y_test.values.ravel(), y_pred=model.predict(X_test)):.4f}")
     print("----------------\n")

     
     # ----- Plotting Loss Curves -----
     if(plot == 1):
         plt.style.use("ggplot")
         plt.plot(model.loss_curve_)
         plt.xlabel("Iteration")
         plt.ylabel("Loss")
         plt.suptitle("Loss Values vs. Iterations")
         plt.show()

default_ANN(plot=1)





# ============================================================================= 
# ====== 3.2 Hyperparameter Tuning ======
# =============================================================================
# ----- Creating a function to create sets of hidden layer tuples -----

def set_layers_array(first, second):

    layers_array = []

    if (second == 0):
        for x in range(1, first):
            layers_array.append((x,))
    else:
        for x in range(1, first):
            for y in range(1, second):
                layers_array.append((x, y))

    return layers_array

# =============================================================================
#  ======= Grid Search =======
# =============================================================================

 #----- Setting parameters range -----
layers_list = set_layers_array(28, 0)

learning_rate_init_list = [0.01, 0.001]
max_iter_list = [300, 350]
 # ----- GridSearch Function -----
mlp_gs = MLPClassifier(max_iter=700)
parameter_space = {
    'hidden_layer_sizes': layers_list ,
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'learning_rate_init': learning_rate_init_list,
}

clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=10)
clf.fit(X_train, y_train.values.ravel()) # X is train samples and y is the corresponding labels
print('Best parameters found:\n', clf.best_estimator_)
best_model = clf.best_estimator_
train_preds = best_model.predict(X_train)
print("Train accuracy: ", accuracy_score(y_true=y_train, y_pred=train_preds))
test_preds = best_model.predict(X_test)
print("Test accuracy: ", accuracy_score(y_true=y_test, y_pred=test_preds))

# =============================================================================
#  ======= Random Search =======
# =============================================================================

 # ----- Setting parameters range -----

layers_list = set_layers_array(28, 28)
print(layers_list)
activation_list = ['identity', 'logistic', 'tanh', 'relu']
learning_rate_init_list = [0.01, 0.001]
c = y_train.values.ravel()
 # ----- RandomSearch Function -----

 # ----- RandomSearch -----
param_grid = {'hidden_layer_sizes': layers_list,
               'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'learning_rate_init': learning_rate_init_list}

random_search = RandomizedSearchCV(MLPClassifier(random_state=42, max_iter=500, early_stopping=False),
                                    param_distributions=param_grid, cv=10,
                                    n_iter=100,
                                    refit=True)

random_search.fit(X_train,y_train.values.ravel())
best_model = random_search.best_estimator_

print('\n-------------')
print(random_search.best_params_)
print('------------- \n')

train_preds = best_model.predict(X_train)
print("Train accuracy: ", accuracy_score(y_true=y_train, y_pred=train_preds))
test_preds = best_model.predict(X_test)
print("Test accuracy: ", accuracy_score(y_true=y_test, y_pred=test_preds))




# =============================================================================
# Manual CV
# =============================================================================
 # # ----- Manual Cross Validation -----
 # # ----- Used to be able to show accuracy on the validation set -----

layers_list = set_layers_array(28, 0)
activation_list = ['identity', 'logistic', 'tanh', 'relu']

kfold = KFold(n_splits=10, shuffle=True, random_state=123)
res = pd.DataFrame()
k = 0
for train_idx, val_idx in kfold.split(X_train):
    k = k + 1
    for layer_size in layers_list:
        print(f"Now training: k = {k}, layer_size = {layer_size}")
        model = MLPClassifier(random_state=42, activation='identity', verbose=False,
                              hidden_layer_sizes=layer_size,
                              early_stopping=True,
                              max_iter=1000, learning_rate_init=0.01)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx].values.ravel())
        train_acc = accuracy_score(y_true=y_train.iloc[train_idx].values.ravel(), y_pred=model.predict(X_train.iloc[train_idx]))
        val_acc = accuracy_score(y_true=y_train.iloc[val_idx].values.ravel(), y_pred=model.predict(X_train.iloc[val_idx]))
        res = res.append({'layer_size': layer_size,
                           'train_acc': train_acc,
                           'val_acc': val_acc}, ignore_index=True)


 # print(res[['max_depth', 'criterion', 'max features', 'train_acc', 'val_acc']].groupby(['max_depth']).mean().reset_index().sort_values('val_acc', ascending=False).head(20))
 # print(res[['max_depth', 'criterion', 'max features', 'train_acc', 'val_acc']].groupby(['max_depth']).std().reset_index().sort_values('val_acc', ascending=False).head(20))


a = res[['layer_size', 'train_acc', 'val_acc']].groupby(['layer_size']).mean().reset_index().sort_values('layer_size', ascending=False)
b = res[['layer_size', 'train_acc', 'val_acc']].groupby(['layer_size']).std().reset_index().sort_values('layer_size', ascending=False)

 # print(a)
 # print(b)
b = b.rename(columns={'train_acc': 'train_std', 'val_acc': 'val_std'})
b = b.drop('layer_size', 1)

result = pd.concat([a, b], axis=1, sort=False)
r = result.sort_values('val_acc', ascending=False).head(10)
#r.to_excel(f"C:\Users\Owner\Desktop\ML\results.csv")

# =============================================================================
# best_config_ANN
# =============================================================================
scaler = StandardScaler()  
X_train_standard = X_train.copy()
columns_to_norm = ['NA_Sales', 'JP_Sales', 'Other_Sales','Critic_Score','User_Score','User_many','User_avg']
X_train_standard[columns_to_norm] = scaler.fit_transform(X_train[columns_to_norm])
model = MLPClassifier(random_state=42, activation='tanh', verbose=False,
                              hidden_layer_sizes=(12,),
                              early_stopping=True,
                              max_iter=1000, learning_rate_init=0.01)
model.fit(X_train_standard, y_train.values.ravel())

# =============================================================================
# 2 classes vs 3 classes
# =============================================================================
data_3class = pd.read_csv(r"C:\Users\Owner\Desktop\ML\חלק ב\data_3_class.csv")
dt_x3 = data_3class.iloc[:,:28].copy()
dt_y3 = pd.DataFrame(data_3class["EU_SALES"]).copy()
lab_enc = preprocessing.LabelEncoder()
dt_y3 = lab_enc.fit_transform(dt_y3)
dt_y3 = pd.DataFrame(dt_y3)
columns_to_norm = ['NA_Sales', 'JP_Sales', 'Other_Sales','Critic_Score','User_Score','User_many','User_avg']
dt_x3[columns_to_norm] = scaler.fit_transform(dt_x3[columns_to_norm])
X_train3, X_test3, y_train3, y_test3 = train_test_split(dt_x3, dt_y3, test_size=0.2 , random_state = 42)
scaler = StandardScaler()  
X_train_standard = X_train3.copy()
# =============================================================================
#  ======= Grid Search for 3 classes =======
# =============================================================================

 #----- Setting parameters range -----
layers_list = set_layers_array(28, 0)

learning_rate_init_list = [0.01, 0.001]
max_iter_list = [300, 350]
 # ----- GridSearch Function -----
mlp_gs = MLPClassifier(max_iter=700)
parameter_space = {
    'hidden_layer_sizes': layers_list ,
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'learning_rate_init': learning_rate_init_list,
}

clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=10)
clf.fit(X_train3, y_train3.values.ravel()) # X is train samples and y is the corresponding labels
print('Best parameters found:\n', clf.best_estimator_)
best_model = clf.best_estimator_
train_preds = best_model.predict(X_train3)
print("Train accuracy: ", accuracy_score(y_true=y_train3, y_pred=train_preds))
test_preds = best_model.predict(X_test3)
print("Test accuracy: ", accuracy_score(y_true=y_test3, y_pred=test_preds))

# =============================================================================
# # Comparison between models
# =============================================================================
prdict_y = pd.DataFrame(model.predict(X_test), columns = ['prdict_y'])
print(confusion_matrix(y_true = y_test, y_pred = prdict_y))
TP = confusion_matrix(y_true = y_test, y_pred = prdict_y)[0,0]
FP = confusion_matrix(y_true = y_test, y_pred = prdict_y)[0,1]
FN = confusion_matrix(y_true = y_test, y_pred = prdict_y)[1,0]
TN = confusion_matrix(y_true = y_test, y_pred = prdict_y)[1,1]
#True Positive Rate (TPR) or Hit Rate or Recall or Sensitivity = TP / (TP + FN)
TPR = TP / (TP + FN)
print("TPR:" +str(TPR))
#False Positive Rate(FPR) or False Alarm Rate = 1 - Specificity = 1 - (TN / (TN + FP))
FPR = 1 - (TN / (TN + FP))
print("FPR:" + str(FPR))
Precision = TP / (TP + FP)
print("Precision:" + str(Precision))
accuracy_com = (TP+TN) / (TP + TN +FN +FP) 
print('accuracy : %.3f' % accuracy_com)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, prdict_y)
auc = roc_auc_score(y_test, prdict_y)
print('AUC: %.3f' % auc)
