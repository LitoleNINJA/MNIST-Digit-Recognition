# Necessary Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale

# Import data
train_orig = pd.read_csv('mnist_train.csv')
test_orig = pd.read_csv('mnist_test.csv')

# Slicing data to resonable size 
train = train_orig[:30000]
test = test_orig[:5000]

# Separating X and y variables
y = train['label']
X = train.drop(['label'], axis=1)

# Normalizing and scaling data
X = X/255.0                             # as 255 is max pixel value
test = test/255.0
X_scaled = scale(X)

# Train - Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=10)


# Logistic Regressor
clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_true=y_test, y_pred=pred), "\n")


# Decision tree
clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)
pred_tree = clf_tree.predict(X_test)
print("Accuracy:", accuracy_score(y_true=y_test, y_pred=pred_tree), "\n")


# Linear Support Vector Machine
clf_svm = SVC(kernel='linear')
clf_svm.fit(X_train, y_train)
pred_svm_lin = clf_svm.predict(X_test)
print("Accuracy:", accuracy_score(y_true=y_test, y_pred=pred_svm_lin), "\n")


# Non-Linear Support Vector Machine
clf_svm = SVC(kernel='rbf', C=10, gamma=0.001)
clf_svm.fit(X_train, y_train)
pred_svm_nlin = clf_svm.predict(X_test)
print("Accuracy:", accuracy_score(y_true=y_test, y_pred=pred_svm_nlin), "\n")


# Plot 10 random images from the test set and check prediction
for i in (np.random.randint(0,1000,10)):
    two_d = (np.reshape(X_test[i], (28, 28)) * 255)
    plt.title('predicted label: {0}'. format(pred_svm_nlin[i]))
    plt.imshow(two_d, cmap='gray')
    plt.show()
