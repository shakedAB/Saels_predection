# -*- coding: utf-8 -*-


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
