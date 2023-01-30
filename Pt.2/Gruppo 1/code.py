##########################################################
#################### IMPORT LIBRARIES ####################
##########################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set seaborn style
sns.set()

##########################################################
##################### DATASET LOADING ####################
##########################################################

# Load training data
mat = scipy.io.loadmat('train.mat')  # Load train.mat dataset into variable mat
data = np.array(mat['train'])  # Create a numpy array from the data in mat
# Assign all rows and all columns except the last one to X_train
X_train = data[:, 0: data.shape[1]-1]
y_train = data[:, [-1]]  # Assign the last column to y_train
y_train = y_train.reshape(-1)  # Reshape y_train into a 1-dimensional array

# Load test data
# Load test.mat dataset into variable mat_test
mat_test = scipy.io.loadmat('test.mat')
# Create a numpy array from the data in mat_test
data_test = np.array(mat_test['test'])
# Assign all rows and all columns except the last one to X_test
X_test = data_test[:, 0: data_test.shape[1]-1]
y_test = data_test[:, [-1]]  # Assign the last column to y_test
y_test = y_test.reshape(-1)  # Reshape y_test into a 1-dimensional array

# Print shapes of X_train, y_train, X_test, and y_test
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Center the data by subtracting the mean from each feature (for both train and test data)
X_train = X_train - np.mean(X_train, axis=0)
X_test = X_test - np.mean(X_test, axis=0)

# Standardize the features (mean 0, var 1)
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

##########################################################
################### CORRELATION MATRIX ###################
##########################################################
'''
    Summary of this block:
    This code creates a dataframe from X values of the scaled train set.
    It then plots a heatmap of the correlation matrix of the dataframe.
'''

# Create matrix of the scaled data
df = pd.DataFrame.from_dict({'X1': X_train_scale[:, 0], 'X2': X_train_scale[:, 1],
                             'X3': X_train_scale[:, 2], 'X4': X_train_scale[:, 3],
                             'X5': X_train_scale[:, 4], 'X6': X_train_scale[:, 5],
                             'X7': X_train_scale[:, 6], 'X8': X_train_scale[:, 7],
                             'X9': X_train_scale[:, 8], 'X10': X_train_scale[:, 9],
                             'X11': X_train_scale[:, 10], 'X12': X_train_scale[:, 11],
                             'X13': X_train_scale[:, 12], 'X14': X_train_scale[:, 13],
                             'X15': X_train_scale[:, 14], 'X16': X_train_scale[:, 15],
                             'X17': X_train_scale[:, 16], 'X18': X_train_scale[:, 17],
                             'X19': X_train_scale[:, 18], 'X20': X_train_scale[:, 19]})

# Plot the heatmap of the correlation matrix
plt.figure("Correlation matrix on train data scaled")
sns.heatmap(df.corr())

##########################################################
####################### PERFORM PCA ######################
##########################################################
'''
    Summary of this block:
    This code performs PCA on the X_train_scale data, keeping all the components, and then creates a DataFrame from the
    transformed X_pca_scale data. The correlation matrix of the X_pca_scale data is then displayed using a heatmap.
'''
# Perform PCA without specifying the n_components parameter, which means all components will be kept
pca_scale = PCA()

# Fit the PCA model to the X_train_scale data
pca_scale.fit(X_train_scale)

# Transform X_train_scale into a new array X_pca_scale by multiplying it with the PCA components
X_pca_scale = pca_scale.transform(X_train_scale)

# Create a DataFrame from the X_pca_scale array and display the correlation matrix for the transformed data after PCA
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

plt.figure("Correlation matrix after PCA")
sns.heatmap(dft.corr())

##########################################################
################ DIMENSIONALITY REDUCTION ################
##########################################################
'''
    Summary of this block:
    This code is creating a plot of the proportion of variance explained by each principal component
    and the cumulative variance explained by all principal components respectively. The plot is showing that 14 principal
    components allow to preserve around 90% of the total data set variance. Therefore, the number of components is set
    to 14 and PCA is applied again to reduce the number of features in the dataset. The original shape and transformed shape
    of the data are also printed to check the dimensionality reduction.
'''
# Create an array with index values from 1 to shape of X
idx = np.arange(X_train_scale.shape[1])+1

# Calculate the proportion of variance explained by each PC and the cumulative variance explained
df_explained_variance_scale = pd.DataFrame([pca_scale.explained_variance_ratio_,
                                            np.cumsum(pca_scale.explained_variance_ratio_)],
                                           index=[
                                               'Proportion of variance explained', 'cumulative'],
                                           columns=idx).T

# print(df_explained_variance_scale)

# Plot the proportion of variance explained and cumulative variance explained
fig, ax1 = plt.subplots(
    num='Proportion of variance explained by each principal component')
ax1.set_xlabel('Principal component', fontsize=14)  # x-axis label
ax1.set_ylabel('Proportion of variance explained', fontsize=14)  # y-axis label

# Plot the proportion of variance explained by each PC
ax2 = sns.barplot(x=idx, y='Proportion of variance explained',
                  data=df_explained_variance_scale, palette='rocket_r')
ax2 = ax1.twinx()
ax2.grid(False)
ax2.set_ylabel('Cumulative variance explained', fontsize=14)  # y-axis label

# Plot the cumulative variance explained by all PCs
ax2 = sns.lineplot(x=idx-1, y='cumulative',
                   data=df_explained_variance_scale, color='k')

# Perform dimensionality reduction using PCA with n_components=16
m = 14
pca_reduced = PCA(n_components=m)
pca_reduced.fit(X_train_scale)
X_pca_reduced = pca_reduced.transform(X_train_scale)
X_pca_test = pca_reduced.transform(X_test_scale)

# Print the shape of the original and transformed datasets
print("original shape:   ", X_train_scale.shape)
print("transformed shape:", X_pca_reduced.shape)


##########################################################
######## DATA VISUALIZATION BEFORE AND AFTER PCA #########
##########################################################
'''
    summary of this block:
    This code is creating two figures, "AFTER PCA" and "BEFORE PCA".
    - In the first figure, it's plotting a scatter plot of X_pca_reduced with the first column on the x-axis
    and the second column on the y-axis, and the color of each point is determined by the corresponding
    value in the y_train list (red if the value is 1, blue if the value is -1).
    - In the second figure, it's plotting a scatter plot of X_train_scale with the first column on the x-axis
    and the second column on the y-axis, and the color of each point is determined by the corresponding
    value in the y_train list (red if the value is 1, blue if the value is -1).
    Both figures are labeled with the x-axis labeled "PC1" and the y-axis labeled "PC2".
'''
# Set colors for each label in y_train
colors = ['red' if label == 1 else 'blue' for label in y_train]


plt.figure("Data visualization after PCA")
# Plot the reduced data with PCA in 2D space
plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=colors)
plt.xlabel("PC1", fontsize=14)  # Label x-axis as PC1
plt.ylabel("PC2", fontsize=14)  # Label y-axis as PC2

plt.figure("Data visualization before PCA")
# Plot the data before PCA in 2D space
plt.scatter(X_train_scale[:, 0], X_pca_scale[:, 1], c=colors)
plt.xlabel("PC1", fontsize=14)  # Label x-axis as PC1
plt.ylabel("PC2", fontsize=14)  # Label y-axis as PC2


##########################################################
################### GRADIENT DESCENT #####################
##########################################################
def sgd(m, x_train, y_train, lr=0.0001, epochs=1000):
    '''
    The function sgd performs stochastic gradient descent on the input data x_train and corresponding labels y_train,
    using a learning rate lr and a specified number of epochs.
    The function initializes a random approximation of the beta parameters and iteratively updates the beta values
    based on the gradient of the cost function for each input example. After the specified number of epochs, the
    function returns a dictionary containing the final beta values for each class.
    Args:
        m (int): represents the number of features in the input data
        x_train (np.array): input data
        y_train (np.array): corresponding labels of x_train
        lr (float, optional): is the learning rate. Defaults to 0.001.
        epochs (int, optional): number of ephocs. Defaults to 100.

    Returns:
        dict: is the final beta values for each class
    '''
    # Initialize the approximation of regression coefficients beta with a normal distribution
    beta_approx = np.random.normal(size=m)
    # Dictionary to store the final values of beta after the last iteration
    beta_dict = dict()
    # List to store the loss for each iteration
    loss_history = []
    # Loop over the number of epochs
    for i in range(epochs):
        j = 1  # Initialize index for beta_dict
        # Loop over the training data and target variables
        for x_i, y_i in zip(x_train, y_train):
            # Calculate gradient of the cost function
            grad = (-2*x_i.T) * (y_i - np.dot(x_i.T, beta_approx))
            beta_approx -= lr*grad  # Update beta approximation using the calculated gradient
            # Store the final values of beta in beta_dict after the last iteration
            if (i == epochs-1):
                beta_dict[j] = beta_approx
                j += 1

        # Calculate the loss using mean squared error (MSE)(tra la y vera e quella predetta calcolata dal prodotto della x e dei beta)
        loss = np.mean((y_train - np.dot(x_train, beta_approx))**2)

        loss_history.append(loss)
    # Return the final values of beta stored in beta_dict and the loss history
    return beta_dict, loss_history


lr = 0.00001
epochs = 100
B_stimato, loss = sgd(m, X_pca_reduced, y_train, lr=lr, epochs=epochs)

plt.figure("Gradient descent, lr=" + str(lr) + ", epochs=" + str(epochs))
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss history, lr=" + str(lr) + ", epochs=" + str(epochs))

##########################################################
###################### CLASSIFIER ########################
##########################################################


def classifier(X, y, Beta):
    '''
    The function implements a binary classifier. The input parameters are:

        X: input data to be classified, with dimensions (num_samples, num_features).
        y: corresponding true class labels of the input data, with dimensions (num_samples,).
        Beta: dictionary of weight vectors for each class, with the class label as the key.

    The function initializes an empty array out_classifier with int type to store the output classifier.
    The function then loops over the rows of the input data X. For each row j, it calculates the dot product
    of the j-th row of X and the transpose of the corresponding weight vector from Beta using y[j] as the key.
    If the dot product is greater than 0, the j-th element of the out_classifier array is assigned 1, else it is assigned -1.
    The function returns the out_classifier array.
    '''
    # Initialize an empty array with int type to store the output classifier
    out_classifier = np.zeros((X.shape[0],), int)
    for j in range(X.shape[0]):  # Iterate over the rows of the input data
        # Dot product of the j-th row of X and the transpose of the corresponding weight vector from Beta, if it's greater than 0
        if (np.dot(X[j], np.transpose(Beta.get(int(y[j])))) > 0):
            # Assign 1 to the j-th element of the out_classifier array
            out_classifier[j] = 1
        else:
            # Assign -1 to the j-th element of the out_classifier array
            out_classifier[j] = -1
    return out_classifier  # Return the out_classifier array


# call the function and store the output to the out_classifier
out_classifier = (classifier(X_pca_test, y_test, B_stimato))


##########################################################
################### ASCII CONVERTION #####################
##########################################################

def bits_to_ascii(bits):
    '''
    The function bits_to_ascii converts a binary array bits to an ASCII string.
    The input parameter bits is an array of binary digits (0 or 1).

    The function initializes an empty string ascii_string to store the final ASCII string.
    It then loops over the input binary array bits in increments of 8. For each iteration,
    the function takes 8 binary digits from bits and converts them to a string byte_str.
    It then converts the byte_str string to an ASCII character using the chr function and the int function with base 2.
    The ASCII character is then added to the ascii_string string.

    Finally, the function returns the ascii_string string, which represents the input binary array bits in ASCII format.
    '''
    ascii_string = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        byte_str = ''.join(str(bit) for bit in byte)  # Convert array to string
        ascii_string += chr(int(byte_str, 2))  # convert binary string to ascii
    return ascii_string


# Print the ascii string
clue = bits_to_ascii(np.where(out_classifier == -1, 0, 1))
print("\n--------------------------\nThe clue is: ",
      clue, "\n--------------------------\n")

plt.show()  # Show the plot
