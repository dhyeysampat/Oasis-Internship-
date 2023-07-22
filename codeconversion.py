import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

# Load the data
vacsv = pd.read_csv("vacation.csv", check_names=False)
vac = pd.DataFrame(vacsv)

# Data exploration
print(vac.columns)
print(vac.shape)
print(vac.iloc[:, [0, 1, 3, 4]].describe())
print(vac['Income2'].value_counts())

# Data preparation
inc2 = vac['Income2']
lev = pd.unique(inc2)[[0, 2, 3, 4, 1]]
inc2 = pd.Categorical(inc2, categories=lev, ordered=True)
print(pd.crosstab(vac['Income2'], inc2))
vac['Income2'] = inc2

# Data visualization
sns.histplot(vac['Age'], kde=False)
plt.show()

sns.histplot(vac['Age'], bins=50, kde=True)
plt.show()

sns.boxplot(x=vac['Age'], orient='horizontal')
plt.xlabel('Age')
plt.show()

# Binary variables
yes_percent = 100 * (vac.iloc[:, 13:33] == "yes").mean()
plt.figure(figsize=(8, 5))
plt.bar(yes_percent.index, yes_percent.values)
plt.xlabel('Activity')
plt.ylabel('Percent "yes"')
plt.title('Percentage of "yes" for each activity')
plt.show()

print(vac['Income'].value_counts().sort_index())
print(vac['Income2'].value_counts())

# Clustering the binary variables
vacmot = (vac.iloc[:, 13:33] == "yes").astype(int)
vacmot_scaled = scale(vacmot)
pca = PCA()
vacmot_pca = pca.fit_transform(vacmot_scaled)

# K-means clustering for binary variables
num_clusters = range(1, 11)
kmeans_models = [KMeans(n_clusters=k, random_state=123) for k in num_clusters]
inertia = [model.fit(vacmot_pca).inertia_ for model in kmeans_models]
plt.plot(num_clusters, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-cluster Sum of Squares (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Silhouette Score for K-means clustering
silhouette_scores = [silhouette_score(vacmot_pca, model.labels_) for model in kmeans_models]
plt.plot(num_clusters, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()

# Hierarchical clustering for binary variables
linkage_matrix = linkage(vacmot_scaled, method='complete', metric='euclidean')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=vacmot.index, leaf_rotation=90)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Observation Index')
plt.ylabel('Distance')
plt.show()

# K-means clustering with flexclust package
PF3 = priceFeature(500, which="3clust")
PF3_kmeans = cclust(PF3, k=3)
print(PF3_kmeans)

# Stepwise clustering with flexclust package
PF3_kmeans_step = stepFlexclust(PF3, k=2, nrep=10)
print(PF3_kmeans_step)

# Mclust for binary variables
PF3_mclust = Mclust(PF3, G=2)
print(PF3_mclust)

# Data for flexclust examples
data("vacmot", package="flexclust")
vacmet = vacmot.iloc[:, [1, 2, 3]]
vacmet = vacmet.dropna()
vacmet = vacmet.astype('int')

# Visualization with pairs plot
sns.pairplot(vacmet, diag_kind='kde')
plt.show()

# Mclust for the vacmet dataset
vacmet_mclust = Mclust(vacmet, G=1:8)
print(vacmet_mclust)

# Data for winterActiv example
data("winterActiv", package="MSA")
winterActiv2 = winterActiv.iloc[:, [1, 7]]
activity_counts = winterActiv2.apply(pd.value_counts).T
activity_counts = activity_counts.div(activity_counts.sum(axis=1), axis=0)
activity_counts.columns = ['No', 'Yes']
print(activity_counts)

# Binary logistic regression with flexmix
winterActiv2_m2 = flexmix(as.formula("~ 1"), k=2, model=FLXMCmvbinary(), data=winterActiv2)
winterActiv2_m14 = stepFlexmix(as.formula("~ 1"), k=1:4, model=FLXMCmvbinary(), nrep=10, verbose=False)
print(winterActiv2_m14)

# Logistic regression with flexmix
best_winterActiv2_m14 = getModel(winterActiv2_m14)
p = parameters(best_winterActiv2_m14)
pi = prior(best_winterActiv2_m14)
expected = np.outer((1 - p[1]), (1 - p[2])) + np.outer(p[1], p[2])
predicted_counts = np.round(n * (pi[1] * expected[:, 0] + pi[2] * expected[:, 1]))
print(predicted_counts)

# Clustering with flexclust for winterActiv
set.seed(1234)
winter_m28 = stepFlexclust(as.formula("~ 1"), k=2:8, nrep=10, model="FLXMCmvbinary", verbose=False)
print(winter_m28)
winter_m5 = getModel(winter_m28, "5")
print(winter_m5)

# Visualization of clustered data
propBarchart(winterActiv, clusters(winter_m5), alpha=1, strip_prefix="Segment ")


# Clustering with Neural Gas in flexclust
data("vacmot", package="flexclust")
set.seed(1234)
vacmot_som <- som(vacmot)
vacmot_ng <- cclust(vacmot_som, k=6, control=list(mincriterion=0.99, members=list(nc=table(vacmot))))
print(vacmot_ng)

# Biclusternumber and visualization
load("ausact-bic.RData")
library("biclust")
bcn <- biclusternumber(ausact.bic)
data("ausActiv", package="MSA")
cl12 <- rep(NA, nrow(ausActiv))
for (k in seq_along(bcn)) {
    cl12[bcn[[k]]] <- k
}
cl12[is.na(cl12)] <- length(bcn) + 1
propBarchart(ausActiv, cl12, alpha=0.7, strip_prefix="Cluster ")
