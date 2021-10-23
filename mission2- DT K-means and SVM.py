# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:53:43 2020

@author: יונתן שטרנברג
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#import data

y_train= pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\y_train.csv")
X_train= pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\X_train.csv")


numOfSamples = X_train.shape[0]
print("Number of samples: " + str(numOfSamples))
print()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        rounded = round(height,5)
        ax.text(x=rect.get_x() + rect.get_width()/2, y=1.050*height, s=rounded, ha='center', va='bottom')


#######'y_train'################################################################


sns.distplot(y_train['EU_Sales'], hist=False, kde=True)
plt.show()


mean = y_train['EU_Sales'].mean()
print(mean)

std  = y_train['EU_Sales'].std()
print(std)

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
# בדיקת מתאם  הפיצרים שהחטנו להשאיר
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
'''
#Dimensionality Reduction
#pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=2)
x = data.drop("EU_SALES", 1)
x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

principalDf.to_excel(r'C:\mlproject2\data.xlsx', index = False)
'''



####################### PART 2  ################

##2.1#####



vec=list(data.columns.values)
print(vec)
#######  Model Training
from sklearn.preprocessing import KBinsDiscretizer 
from matplotlib import pyplot

# discretization transform the raw data
kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
data["EU_SALES"] = kbins.fit_transform(data[["EU_SALES"]])

#checking
pyplot.hist(data["EU_SALES"])
pyplot.show()
# end checking


data["EU_SALES"] = data["EU_SALES"].fillna(0)

y_train_handled = pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\y_train_handled.csv")
X_train_handled= pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\X_train_handled.csv")

# =============================================================================
# #------------- Decision Trees
# =============================================================================

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split



model = DecisionTreeClassifier(criterion='entropy')
lab_enc = preprocessing.LabelEncoder()

X_train_handled = X_train_handled.iloc[:,:28]
y_train_handled = pd.DataFrame(y_train_handled["EU_SALES"])
y_train_handled = lab_enc.fit_transform(y_train_handled)
y_train_handled = pd.DataFrame(y_train_handled)

## split to val and train ######
X_train, X_test, y_train, y_test = train_test_split(X_train_handled, y_train_handled, test_size=0.2, random_state=123)
print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
print(f"Train size: {y_train.shape[0]}")
print(f"Test size: {y_test.shape[0]}")


max_depth_list = np.arange(1, 25, 1)
print(max_depth_list)


#####graph for val and trainig set#####
res1 = pd.DataFrame()
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    res1 = res1.append({'max_depth': max_depth,
                      'train_acc':accuracy_score(y_train, model.predict(X_train)),
                      'val_acc':accuracy_score(y_test, model.predict(X_test))}, ignore_index=True)
    
    
plt.figure(figsize=(13, 4))
plt.ylim((0.7,1))
plt.plot(res1['max_depth'], res1['train_acc'], marker='o', markersize=4)
plt.plot(res1['max_depth'], res1['val_acc'], marker='o', markersize=4)
plt.legend(['Train accuracy', 'Test accuracy'])
plt.show()


model.fit(X_train, y_train)
plt.figure(figsize=(12, 10))
plot_tree(model, filled=True, class_names=True)
plt.show()

vec=list(data.columns.values)
print(vec)

##print only 2 levels##

plt.figure(figsize=(12, 10))
plot_tree(model, filled=True,max_depth=2, class_names=['0', '1'], feature_names=['NA_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'User_Score', 'Platform_GC', 'Platform_PC', 'Platform_PS', 'Platform_PS3', 'Platform_PS4', 'Platform_Wii', 'Year_of_Release_1997', 'Year_of_Release_2002', 'Year_of_Release_2003', 'Year_of_Release_2013', 'Year_of_Release_2014', 'Genre_Fighting', 'Genre_Racing', 'Genre_Role-Playing', 'Publisher_GT Interactive', 'Publisher_Microsoft Game Studios', 'Publisher_RedOctane', 'Publisher_Russel', 'Publisher_Square Enix', 'Publisher_SquareSoft', 'Publisher_Take-Two Interactive', 'User_many', 'User_avg'], fontsize=10)
plt.show()

##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.2f}")
print(model.tree_.max_depth)

##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test)):.2f}")


##K- Fold###

##split to trin and set data##


kfold = KFold(n_splits=10, shuffle=True, random_state=123)
res = pd.DataFrame()
k = 0
for train_idx, val_idx in kfold.split(X_train):
    for max_depth in max_depth_list:
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        acc = accuracy_score(y_train.iloc[val_idx], model.predict(X_train.iloc[val_idx]))
        res = res.append({'max_depth': max_depth,
                          'k':k,
                          'acc': acc}, ignore_index=True)
        
        
print(res[['max_depth', 'acc']].groupby(['max_depth']).mean().reset_index().sort_values('acc', ascending=False).head(5))
print(res[['max_depth', 'acc']].groupby(['max_depth']).std().reset_index().sort_values('acc', ascending=False).head(10))


##דוגמה ספצפית לאימון וואלידציה - לקחנו k ספיצפי כדי להציג את זה בגרף
X_train, X_val, y_train, y_val = train_test_split(dt_x, dt_y, test_size=0.2, random_state=123)
res = pd.DataFrame()
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    res = res.append({'max_depth': max_depth,
                      'train_acc':accuracy_score(y_train, model.predict(X_train)),
                      'val_acc':accuracy_score(y_val, model.predict(X_val))}, ignore_index=True)
plt.figure(figsize=(13, 4))
plt.plot(res['max_depth'], res['train_acc'], marker='o', markersize=4)
plt.plot(res['max_depth'], res['val_acc'], marker='o', markersize=4)
plt.legend(['Train accuracy', 'Validation accuracy'])
plt.show()


# ====== 2.2 Hyperparameter Tuning ======

from sklearn.model_selection import GridSearchCV

    # ----- GridSearch -----

param_grid = {'max_depth': np.arange(1, 20, 1),
              'criterion': ['entropy', 'gini'],
              'max_features': ['auto', 'sqrt', 'log2', None]
              }

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
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

# code of shaked 2.2####


# # ----- Manual Cross Validation -----
# # ----- Used to be able to show accuracy on the validation set -----

max_depth_list = np.arange(1, 20, 1)
crit_list = ['entropy', 'gini']
max_features_list = ['auto', 'sqrt', 'log2', None]


kfold = KFold(n_splits=10, shuffle=True, random_state=123)
res = pd.DataFrame()
k = 0
for train_idx, val_idx in kfold.split(X_train):
    k = k + 1
    for max_depth in max_depth_list:
        print(f"Now training: k = {k}, max_depth = {max_depth}")
        model = DecisionTreeClassifier(criterion='entropy', max_features=None, max_depth=max_depth, random_state=42)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        train_acc = accuracy_score(y_true=y_train.iloc[train_idx], y_pred=model.predict(X_train.iloc[train_idx]))
        val_acc = accuracy_score(y_true=y_train.iloc[val_idx], y_pred=model.predict(X_train.iloc[val_idx]))
        k += 1
        res = res.append({'max_depth': max_depth,
                          'train_acc': train_acc,
                          'val_acc': val_acc}, ignore_index=True)


# print(res[['max_depth', 'criterion', 'max features', 'train_acc', 'val_acc']].groupby(['max_depth']).mean().reset_index().sort_values('val_acc', ascending=False).head(20))
# print(res[['max_depth', 'criterion', 'max features', 'train_acc', 'val_acc']].groupby(['max_depth']).std().reset_index().sort_values('val_acc', ascending=False).head(20))


a = res[['max_depth', 'train_acc', 'val_acc']].groupby(['max_depth']).mean().reset_index().sort_values('max_depth', ascending=False)
b = res[['max_depth', 'train_acc', 'val_acc']].groupby(['max_depth']).std().reset_index().sort_values('max_depth', ascending=False)

print(a)
print(b)

b = b.rename(columns={'train_acc': 'train_std', 'val_acc': 'val_std'})
b = b.drop('max_depth', 1)

result = pd.concat([a, b], axis=1, sort=False)
result1=result.sort_values('val_acc', ascending=False).head(10)

print(result.sort_values('val_acc', ascending=False).head(10))

result1.to_excel(r'C:\Users\יונתן שטרנברג\Desktop\ML mission 1\outputdtNone.xlsx', index = False, header=True)




#######################


# ====== 2.3 Using Best Config. Models ======

## print the test score after Hyperparameter Tuning####
model = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
preds2 = model.predict(X_train)
print("Test accuracy: ", accuracy_score(y_train, preds2))
print("Test accuracy: ", accuracy_score(y_test, preds))


########################

criterion_config = 'gini'
max_features_config = None
max_depth_config = 8
alpha = 0
analayze_prune=True
feature_importance=True
plot=True

model = DecisionTreeClassifier(criterion=criterion_config,
                                    max_features=max_features_config,
                                    max_depth=max_depth_config,
                                    ccp_alpha=alpha,
                                    random_state=42)

model.fit(X_train, y_train)

# ----- Post Pruning -----


# ----- Finding CCP Values For Post-Pruning -----
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
     
# ----- Total Impurity vs effective alpha for Training set -----

plt.style.use("ggplot")
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs. effective alpha for Training Set")

# ----- Nodes vs. CCP Alpha -----

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clf.tree_.node_count, ccp_alpha))
    
# ----- Nodes vs. CCP Alpha -----

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]

fig, ax = plt.subplots(2, 1)


# ----- Nodes vs. CCP Alpha -----

ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs. alpha")   


# ----- Depth vs. CCP Alpha -----
plt.ylim((25,250))
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs. alpha")
plt.yticks(depth)
fig.tight_layout()


# ----- Accuracy vs Alpha for Test & Validation -----

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs. alpha")
ax.plot(ccp_alphas, train_scores, marker='o', label="Train Set", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="Validation Set", drawstyle="steps-post")
ax.legend(edgecolor='black', shadow=True)
plt.show()


# ----- Plot Tree -----

feature_names = list(X_train.columns)
plt.figure(figsize=(12, 10))
plot_tree(model, filled=True, max_depth=3, class_names=['0', '1'], feature_names=feature_names, fontsize=6)
plt.show()

print("----------------\n")
print(f"Train Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.4f}")
print(f"Test Accuracy: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test)):.4f}")
print("----------------\n")

# ----- Feature Importance -----

df = pd.DataFrame({'col_name': model.feature_importances_},index=X_train.columns).sort_values(by='col_name', ascending=False)
print(df.head(10))


# =============================================================================
# compre DT to other models
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




# =============================================================================
# Clustering
# =============================================================================

#Dimensionality Reduction
#pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering


pca = PCA(n_components=2)
x = X_train.copy()
x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])




pca = PCA(n_components=2)
pca.fit(X_train_handled)

print('\n-------------')
print('Explained Variance - Training Set')
print('-------------')
print(f"PCA 1 Explained Variance: {pca.explained_variance_ratio_[0]:.5f}")
print(f"PCA 2 Explained Variance: {pca.explained_variance_ratio_[1]:.5f}")
print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.5f}")

X_train_pca = pca.transform(X_train_handled)
X_train_pca = pd.DataFrame(X_train_pca, columns=['pc1', 'pc2'])
kmeans = KMeans(n_clusters=2, max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_train_handled)
predictions = pd.DataFrame(kmeans.predict(X_train_handled), columns=['prediction'])
X_train_pca['prediction']=predictions['prediction'].copy()
X_train_pca['y'] = y_train['EU_SALES']
X_train_pca= X_train_pca.dropna(how='any',axis=0)
sns.scatterplot(x='pc1', y='pc2', hue='prediction',data=X_train_pca)
plt.show()

sns.scatterplot(x='pc1', y='pc2', hue='y',data=X_train_pca)
plt.title('Original data Labels')
plt.show()


##Score for the Clustering

matching = np.where(X_train_pca['prediction'] == X_train_pca['y'], 1, 0)
non_matching = np.where(X_train_pca['prediction'] != X_train_pca['y'], 1, 0)
total = X_train_pca.shape[0]

print('\n-------------')
print('Accuracy - Training Set')
print('-------------')
print(f"Total number of samples: {total} ")

print(f"Total number of matching classifications: {matching.sum()}")
print(f"Total number of non-matching classifications: {non_matching.sum()}")

print(f"Train Accuracy for matching: {matching.sum() / total:.5f} ")
print(f"Train Accuracy for non-matching: {non_matching.sum() / total:.5f} ")



# ----- Test Set -----

pca.fit(X_test)

print('\n-------------')
print('Explained Variance - Test Set')
print('-------------')
print(f"PCA 1 Explained Variance: {pca.explained_variance_ratio_[0]:.5f}")
print(f"PCA 2 Explained Variance: {pca.explained_variance_ratio_[1]:.5f}")
print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.5f}")

X_test_pca = pca.transform(X_test)
X_test_pca = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2'])
X_test_pca['y'] = y_test['EU_SALES']

predictions = pd.DataFrame(kmeans.predict(X_test), columns=['prediction'])

X_test_pca['prediction'] = predictions['prediction']



matching = np.where(X_test_pca['prediction'] == X_test_pca['y'], 1, 0)
non_matching = np.where(X_test_pca['prediction'] != X_test_pca['y'], 1, 0)
total = X_test_pca.shape[0]

print('\n-------------')
print('Accuracy - Test Set')
print('-------------')
print(f"Total number of samples: {total} ")

print(f"Total number of matching classifications: {matching.sum()}")
print(f"Total number of non-matching classifications: {non_matching.sum()}")

print(f"Test Accuracy for matching: {matching.sum() / total:.5f} ")
print(f"Test Accuracy for non-matching: {non_matching.sum() / total:.5f} ")

# ----- Plotting -----


datasets = [X_train_pca, X_test_pca]
y_columns = ['y', 'prediction']
marker = 0
for data in datasets:
    marker = marker+1
    for hue in y_columns:
        plt.style.use("ggplot")
        sns.scatterplot(x='pc1', y='pc2', hue=hue, data=data)
        if(hue == 'y'):
            suptitle = "PCA Original Data"
        if(hue == 'prediction'):
            plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
                        pca.transform(kmeans.cluster_centers_)[:, 1],
                        marker='*', s=120, color='Black')
            suptitle = "PCA KMeans Predictions"
        if(marker == 1):
            title = "Train"
        if(marker == 2):
            title = "Test"


        plt.suptitle(suptitle)
        plt.title(title)
        plt.legend(edgecolor='black', shadow=True)
        plt.savefig('sss', bbox_inches='tight')
        plt.show()
        
        
# ----- Plotting -----


datasets = [X_train_pca, X_test_pca]
y_columns = ['y', 'prediction']
marker = 0
for data in datasets:
    marker = marker+1
    for hue in y_columns:
        plt.style.use("ggplot")
        sns.scatterplot(x='pc1', y='pc2', hue=hue, data=data)
        if(hue == 'y'):
            suptitle = "PCA Original Data"
        if(hue == 'prediction'):
            # plt.scatter(pca.transform(spectral.cluster_centers_)[:, 0],
            #             pca.transform(spectral.cluster_centers_)[:, 1],
            #             marker='*', s=120, color='Black')
            suptitle = "PCA SpectralClustering Predictions"
        if(marker == 1):
            title = "Set - Train"
        if(marker == 2):
            title = "Set - Test"


        plt.suptitle(suptitle)
        plt.title(title)
        plt.legend(edgecolor='black', shadow=True)
        plt.savefig('lll', bbox_inches='tight')
        plt.show()




# Run the Kmeans algorithm and get the index of data points clusters
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X_train_handled)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

kmeans = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_train_handled)
predictions = pd.DataFrame(kmeans.predict(X_train_handled), columns=['prediction'])
X_train_pca['prediction']=predictions['prediction'].copy()
sns.scatterplot(x='pc1', y='pc2',hue='prediction' ,data=X_train_pca)
plt.show()


X_train_3_class= pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\data_3_class.csv")
y_train_3_class=pd.DataFrame()
y_train_3_class['y']=X_train_3_class['EU_SALES'].copy()
X_train_3_class.drop('EU_SALES',axis='columns', inplace=True)


pca = PCA(n_components=2)
x = X_train_3_class.copy()
x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])




pca = PCA(n_components=2)
pca.fit(X_train_3_class)

print('\n-------------')
print('Explained Variance - Training Set')
print('-------------')
print(f"PCA 1 Explained Variance: {pca.explained_variance_ratio_[0]:.5f}")
print(f"PCA 2 Explained Variance: {pca.explained_variance_ratio_[1]:.5f}")
print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.5f}")

X_train_pca_3_class = pca.transform(X_train_3_class)
X_train_pca_3_class = pd.DataFrame(X_train_pca_3_class, columns=['pc1', 'pc2'])
kmeans = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_train_3_class)
predictions = pd.DataFrame(kmeans.predict(X_train_3_class), columns=['prediction'])
X_train_pca_3_class['prediction']=predictions['prediction'].copy()
X_train_pca_3_class['y'] = y_train_3_class['y']
X_train_pca_3_class= X_train_pca_3_class.dropna(how='any',axis=0)
sns.scatterplot(x='pc1', y='pc2', hue='prediction',data=X_train_pca_3_class)
plt.show()

sns.scatterplot(x='pc1', y='pc2', hue='y',data=X_train_pca_3_class)
plt.title('Original data Labels')
plt.show()




# =============================================================================
# ## 3- classes
# =============================================================================
from sklearn.preprocessing import KBinsDiscretizer 
from matplotlib import pyplot
from tqdm import tqdm


# discretization transform the raw data
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
data["EU_SALES"] = kbins.fit_transform(data[["EU_SALES"]])

#checking
pyplot.hist(data["EU_SALES"])
pyplot.show()

data_3_class=data.copy()
data_3_class.to_excel(r'C:\Users\יונתן שטרנברג\Desktop\ML mission 1\data_3_class.xlsx', index = False, header=True)

##find the number of clusters#############################################################
iner_list = []
dbi_list = []
sil_list = []

for n_clusters in tqdm(range(2, 10, 1)):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_train_handled)
    assignment = kmeans.predict(X_train_handled)
    
    iner = kmeans.inertia_
    sil = silhouette_score(X_train_handled, assignment)
    dbi = davies_bouldin_score(X_train_handled, assignment)
    
    dbi_list.append(dbi)
    sil_list.append(sil)
    iner_list.append(iner)
    
    
plt.plot(range(2, 10, 1), iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()



# =============================================================================
# ## comparing Adabost performance to DT after hyperparameter Tuning with the same data
# =============================================================================

y_train_handled = pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\y_train_handled.csv")
X_train_handled= pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\X_train_handled.csv")

X_train, X_test, y_train, y_test = train_test_split(X_train_handled, y_train_handled, test_size=0.2, random_state=123)


model = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=42)
model.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_true=y_train, y_pred=model.predict(X_train)):.2f}")
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test, y_pred=model.predict(X_test)):.2f}")



## Random Forest

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = RandomForestClassifier(max_depth=8, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))




X_train, y_train = make_classification(n_samples=4905, n_features=28,
                            n_informative=2, n_redundant=0,
                            random_state=42, shuffle=False)
clf = RandomForestClassifier(max_depth=8, random_state=42)
clf.fit(X_train, y_train)
RandomForestClassifier(...)
print(clf.predict([[0, 0, 0, 0]]))

print(f"Accuracy: {accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)):.2f}")
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)):.2f}")

# =============================================================================
# HistGradientBoostingClassifier
# =============================================================================
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


param_grid = {
                'n_estimators': [100,200,300,400,500,600],
               'learning_rate':[0.5,0.1,1.5,2,2.5],
               'max_depth' : [8,10,12,14,16,18,20,22,24,26]
               }
Grid = GridSearchCV(HistGradientBoostingClassifier(),param_grid = param_grid)
model = HistGradientBoostingClassifier().fit(X_train, y_train.values.ravel())
best_model = model.best_estimator_
print(best_model)
train_preds = model.predict(X_train)
##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=y_train.values.ravel(), y_pred=train_preds):.2f}")
train_preds = model.predict(X_test)
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test.values.ravel(), y_pred=model.predict(X_test)):.2f}")

hgb_pipe = make_pipeline(
                         HistGradientBoostingClassifier())
parameters = {
  'histgradientboostingclassifier__max_iter': [1000,1200,1500],
 'histgradientboostingclassifier__learning_rate': [0.1],
 'histgradientboostingclassifier__max_depth' : [25, 50, 75],
 'histgradientboostingclassifier__l2_regularization': [1.5],
 }
#instantiate the gridsearch
hgb_grid = GridSearchCV(hgb_pipe, parameters, n_jobs=5, 
 cv=5,
 verbose=2, refit=True)
#fit on the grid 
hgb_grid.fit(X_train, y_train.values.ravel())
train_preds = hgb_grid.predict(X_train)
##accuracy for training set##
print(f"Accuracy: {accuracy_score(y_true=y_train.values.ravel(), y_pred=train_preds):.2f}")
train_preds = hgb_grid.predict(X_test)
##accuracy for validation set##
print(f"Accuracy: {accuracy_score(y_true=y_test.values.ravel(), y_pred=model.predict(X_test)):.2f}")

X_test_final = pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\y_train_handled.csv")




