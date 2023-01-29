# %%
# LIBRARIES LOADING

# seaborn is a statistical data visualization library based on matplotlib
# which reminds R visualization tools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io
import numpy.random as npr


sns.set()

# Load training data
mat = scipy.io.loadmat('train.mat')
data = np.array(mat['train'])
X = data[:, 0: data.shape[1]-1]
y = data[:, [-1]]
y = y.reshape(-1)

# Load test data
mat_test = scipy.io.loadmat('test.mat')
data_test = np.array(mat_test['test'])
X_test = data_test[:, 0: data_test.shape[1]-1]
y_test = data_test[:, [-1]]
y_test = y_test.reshape(-1)

# print("X shape:", X.shape)
# print("y shape:", y.shape)

# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)

# Center the data set removing to each feature its mean
X = X - np.mean(X, axis=0)
X_test = X_test - np.mean(X_test, axis=0)

# Normalize data
# X_scale, X_test_scale = Utils.normalize(X, X_test)
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
X_test_scale = scaler.fit_transform(X_test)
# print(X_scale)

# %%
# CORRELATION MATRIX OF THE ORIGINAL DATA
# To better visualize the data features dependencies let us compute the data set
# correlation matrix
df = pd.DataFrame.from_dict({'X1': X_scale[:, 0], 'X2': X_scale[:, 1],
                             'X3': X_scale[:, 2], 'X4': X_scale[:, 3],
                             'X5': X_scale[:, 4], 'X6': X[:, 5],
                             'X7': X_scale[:, 6], 'X8': X_scale[:, 7],
                             'X9': X_scale[:, 8], 'X10': X_scale[:, 9],
                             'X11': X_scale[:, 10], 'X12': X_scale[:, 11],
                             'X13': X_scale[:, 12], 'X14': X_scale[:, 13],
                             'X15': X_scale[:, 14], 'X16': X_scale[:, 15],
                             'X17': X_scale[:, 16], 'X18': X_scale[:, 17],
                             'X19': X_scale[:, 18], 'X20': X_scale[:, 19]})
# #print(df.corr())
plt.figure("Matrice di correlazione")
sns.heatmap(df.corr())

# %%
# PCA WITH SKLEARN

# We now use the sklearn implementation of PCA. sklearn PCA requires as input
# the number of required principal components, for the moment we do not set any
# value so all the components are retained and we can compare its results with
# our eigendecomposition/SVD implementation. Note that by default sklearn PCA
# uses SVD.
# Further information about sklearn pca at https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# PCA is sensitive to scaling, in order to avoid that features with large
# variance "obscure" others we normalize the data
pca_scale = PCA()
pca_scale.fit(X_scale)
# the method transform gives the principal components, i.e. computes X*V (X*W)
X_pca_scale = pca_scale.transform(X_scale)
# print("original shape:   ", X_scale.shape)
# print("transformed shape:", X_pca_scale.shape)

# the principal directions and the variance (eigenvectors and eigenvalues of the
# data set covariance matrix) are stored as attributes of the class PCA. Note
# that the principal directions are already sorted by decreasing variance
# variance using scaled data. Be careful, in sklearn principal directions are
# arranged per rows and not columns!
# #print('Variance with scaled data')

var_pca_scale = pd.DataFrame(np.stack((pca_scale.explained_variance_, pca_scale.explained_variance_ratio_), axis=1), columns=[
                             'Eigenvalues', 'Explained variance'])
# print(var_pca_scale)

# as expected, PCA also decorrelates the features in the transformed space
dft = pd.DataFrame.from_dict({'PC1': X_pca_scale[:, 0], 'PC2': X_pca_scale[:, 1],
                              'PC3': X_pca_scale[:, 2], 'PC4': X_pca_scale[:, 3],
                              'PC5': X_pca_scale[:, 4], 'PC6': X_pca_scale[:, 5],
                              'PC7': X_pca_scale[:, 6], 'PC8': X_pca_scale[:, 7],
                              'PC9': X_pca_scale[:, 8], 'PC10': X_pca_scale[:, 9],
                              'PC11': X_pca_scale[:, 10], 'PC12': X_pca_scale[:, 11],
                              'PC13': X_pca_scale[:, 12], 'PC14': X_pca_scale[:, 13],
                              'PC15': X_pca_scale[:, 14], 'PC16': X_pca_scale[:, 15],
                              'PC17': X_pca_scale[:, 16], 'PC18': X_pca_scale[:, 17],
                              'PC19': X_pca_scale[:, 18], 'PC20': X_pca_scale[:, 19]})

# #print(dft)

plt.figure("Heatmap PCA")
sns.heatmap(dft.corr())


# %%
# DIMENSIONALITY REDUCTION

# In order to reduce data dimensionality we have to choose a subset of
# principal components to retain. We can do that by looking at the proporion of
# variance explained.

# the attribute explained_variance_ratio_ of the class PCA stores the percentage
# of total variance associated with each principal component
# Scree plot for dimensionality reduction analysis
idx = np.arange(20)+1
df_explained_variance_scale = pd.DataFrame([pca_scale.explained_variance_ratio_, np.cumsum(pca_scale.explained_variance_ratio_)],
                                           index=[
                                               'Proportion of variance explained', 'cumulative'],
                                           columns=idx).T


fig, ax1 = plt.subplots()
ax1.set_xlabel('Principal component', fontsize=14)
ax1.set_ylabel('Proportion of variance explained', fontsize=14)
ax2 = sns.barplot(x=idx, y='Proportion of variance explained',
                  data=df_explained_variance_scale, palette='rocket_r')
ax2 = ax1.twinx()
ax2.grid(False)
ax2.set_ylabel('Cumulative variance explained', fontsize=14)
ax2 = sns.lineplot(x=idx-1, y='cumulative',
                   data=df_explained_variance_scale, color='k')


# the plot shows that 8 principal components allow to preserve more than
# 90% of the total data set variance! Therefore, we re-apply PCA setting the
# number of components to two
m = 16
pca_reduced = PCA(n_components=m)
pca_reduced.fit(X_scale)
X_pca_reduced = pca_reduced.transform(X)
X_pca_test = pca_reduced.transform(X_test_scale)
# print("original shape:   ", X_scale.shape)
# print("transformed shape:", X_pca_reduced.shape)


# %%
# DATA VISUALIZATION AFTER PCA

colors = ['red' if label == 1 else 'blue' for label in y]

plt.figure("X_pca_reduced")
# Create scatter plot
plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=colors)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)

plt.figure("X_pca_scale")
# Create scatter plot
plt.scatter(X_pca_scale[:, 0], X_pca_scale[:, 1], c=colors)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)

# Show plot

# plt.show()

# %%
# GRADIENT DESCENT
# Calcolo del grandiente per risolvere il problema della regressione lineare multipla.

X_pca_reduced = scaler.fit_transform(X_pca_reduced)
X_pca_test = scaler.fit_transform(X_pca_test)

def predict(x, beta):
    return np.dot(x.T, beta).sum()


def sgd(m, x_train, y_train, lr=0.001, epochs=1):
    beta_approx = np.random.normal(size=m)
    beta_dict = dict()
    for i in range(epochs):
        j = 1
        for x_i, y_i in zip(x_train, y_train):
            grad = (-2*x_i.T) * (y_i - np.dot(x_i.T, beta_approx))
            beta_approx -= lr*grad
            if (i == epochs-1):
                beta_dict[j] = beta_approx
                j += 1
    return beta_dict

B_stimato = sgd(m, X_pca_reduced, y, epochs=1000)

# %%
# TEST

def test(X, y, Beta):
    frase_bin = ""
    for j in range(X.shape[0]):
        if (np.dot(X[j], np.transpose(Beta.get(int(y[j])))) > 0):
            frase_bin = frase_bin + str(1)
        else:
            frase_bin = frase_bin + str(0)
    return frase_bin

def bits_to_ascii(bits):
    ascii_string = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        ascii_string += chr(int(byte, 2))
    return ascii_string

# bits = "101010010111010010101010010101010010101010100111"
# #print(bits_to_ascii(bits))

string = (test(X_pca_test, y_test, B_stimato))
print(bits_to_ascii(string))

# plt.show()
