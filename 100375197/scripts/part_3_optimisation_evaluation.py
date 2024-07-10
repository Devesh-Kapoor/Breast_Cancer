# Description: Optimisation and evaluation of classifiers 
# We will be using the MNIST dataset for this part of the assignment.
# comparing the performance of 3 classifiers: Decision Tree, Random Forest and KNN
# We will then use 10 fold cross-validation with no shuffling to optimise the the most important
# classifiers to see how they perform on seen and unseen daat.
# Created on: 12/12/2023
# Version number: 13.12.23, optimisation and evaluation of classifiers

# Importing libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Importing the MNIST dataset
numbers = datasets.load_digits()

# plot the first 4 images in the dataset
import matplotlib.pyplot as plt

# 10 fold cross validation with no shuffling
seed = 97
kfold = 10

# spliting the dataset into the training set and test set
X = numbers.data
y = numbers.target

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)

# use cross_val_score to evaluate the performance of the classifiers

# Decision Tree
DT = DecisionTreeClassifier(criterion='entropy', max_depth = 10 , max_features = 10 ,random_state=seed)
DT.fit(X_train, y_train)
# 10 fold cross validation with no shuffling
scores = cross_val_score(DT, X, y, cv=kfold, scoring='accuracy')
# mean accuracy 
print("Decision Tree Accuracy: %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std() * 2))

# vary the parameters of the classifier to see if we can improve the performance
# max_depth = 5 , max_features = 5
DT_scores = []
DT_scores_std = []
for i in range(6, 11):
    DT = DecisionTreeClassifier(criterion='entropy', max_depth = i , max_features = 10 ,random_state=seed)
    DT.fit(X_train, y_train)
    scores = cross_val_score(DT, X, y, cv=kfold, scoring='accuracy')
    # mean accuracy 
    print("Decision Tree Accuracy: %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std() * 2))
    DT_scores.append(scores.mean())
    DT_scores_std.append(scores.std() * 2)

# fitting the classifier with the best parameters this however still shows 
#that the classifier is overfitting meaning that it does not see the pattern in the data

DT = DecisionTreeClassifier(criterion='entropy', max_depth = 100 , max_features = 10 ,random_state=seed)
DT.fit(X_train, y_train)
scores = cross_val_score(DT, X, y, cv=kfold, scoring='accuracy')
# mean accuracy
print("Decision Tree Accuracy: %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std() * 2))

# the confusion matrix shows how the classifier is performing
# fitting the classifier into a confusion matrix provides us with a visual 
#representation of how the classifier is fitting the data
y_pred = DT.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
display_CM = metrics.ConfusionMatrixDisplay(confusion_matrix)
display_CM.plot()
plt.show()


# Random Forest
RandomForest = RandomForestClassifier(max_depth=10, random_state=seed, max_samples=10)
RandomForest.fit(X_train, y_train)
scores = cross_val_score(RandomForest, X, y, cv=kfold, scoring='accuracy')
# mean accuracy
print("Random Forest Accuracy: %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std() * 2))
# vary the parameters of the classifier to see if we can improve the performance
RF_scores = []
RF_scores_std = []
for i in range(6, 11):
    RandomForest = RandomForestClassifier(max_depth=10, random_state=seed, max_samples=i)
    RandomForest.fit(X_train, y_train)
    scores = cross_val_score(RandomForest, X, y, cv=kfold, scoring='accuracy')
    # mean accuracy
    print("Random Forest Accuracy: %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std() * 2))
    RF_scores.append(scores.mean())
    RF_scores_std.append(scores.std() * 2)

# fitting the classifier into a confusion matrix
y_pred = RandomForest.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
display_CM = metrics.ConfusionMatrixDisplay(confusion_matrix)
display_CM.plot()
plt.show()


# KNN
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(X_train, y_train)
scores = cross_val_score(KNN, X, y, cv=kfold, scoring='accuracy')
# mean accuracy
print("KNN Accuracy: %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std() * 2))

# vary the parameters of the classifier to see if we can improve the performance
KNN_scores = []
KNN_scores_std = []
for i in range(6, 11):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(X_train, y_train)
    scores = cross_val_score(KNN, X, y, cv=kfold, scoring='accuracy')
    # mean accuracy
    print("KNN Accuracy: %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std() * 2))
    KNN_scores.append(scores.mean())
    KNN_scores_std.append(scores.std() * 2)

# confusion matrix
y_pred = KNN.predict(X_test)
# plot the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
display_CM = metrics.ConfusionMatrixDisplay(confusion_matrix)
display_CM.plot()
plt.show()


# plot the accuracy of the classifiers in comparison to the parameters
# Decision Tree accuracy vs max_depth
#showing the effect of the max_depth on the accuracy of the classifier
plt.figure()
plt.plot(range(6, 11), DT_scores, label='Decision Tree')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Random Forest accuracy vs max_samples
#showing the effect of the max_samples on the accuracy of the classifier
plt.figure()
plt.plot(range(6, 11), RF_scores, label='Random Forest')
plt.xlabel('max_samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# KNN accuracy vs n_neighbors
#showing the effect of the n_neighbors on the accuracy of the classifier
plt.figure()
plt.plot(range(6, 11), KNN_scores, label='KNN')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# making a bar chart to compare the accuracy of the classifiers all on one bar 
br1 = np.arange(len(DT_scores))
br2 = [x + 0.25 for x in br1]
br3 = [x + 0.25 for x in br2]

# plot the accuracy of the classifiers to show the accuracy of the classifiers vs the parameters
# this puts all the classifiers on one bar chart so it is easier to compare them
plt.bar(br1, DT_scores, color='r', width=0.25, label='Decision Tree')
plt.bar(br2, RF_scores, color='g', width=0.25, label='Random Forest')
plt.bar(br3, KNN_scores, color='b', width=0.25, label='KNN')
plt.xlabel('Values ID')
plt.ylabel('Accuracy')
plt.title('Accuracy of Classifiers')
plt.legend(loc = 'lower right')
plt.show()