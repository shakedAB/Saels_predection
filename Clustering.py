# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:27:08 2021

@author: יונתן שטרנברג
"""

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


y_train_handled = pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\y_train_handled.csv")
X_train_handled= pd.read_csv(r"C:\Users\יונתן שטרנברג\Desktop\ML mission 1\X_train_handled.csv")

X_train, X_test, y_train, y_test = train_test_split(X_train_handled, y_train_handled, test_size=0.2, random_state=123)

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
