__author__ = 'jkoyan'

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.cross_validation import train_test_split
#import pandas as pd
import  numpy as np
from StringIO import StringIO


iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Create the Naive Bayes Classifier
clf = GaussianNB()

# Train the classifier using the fit method
clf.fit(X_train,y_train)

# Generate predictions i.e. class names on the test data set
y_predict = clf.predict(X_test)

score = accuracy_score(y_test,y_predict,normalize=False)

print("Total number of correctly classified observations: {0} out of {2} observations, Accuracy of the predictions: {1}").format(score,score/float(len(y_test)),len(y_test))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Compute confusion matrix
cm = confusion_matrix(y_test,y_predict)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)

plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()