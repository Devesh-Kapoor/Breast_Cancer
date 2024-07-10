# Description: Making an ensemble of the 3 decision trees and finding the accuracy of the ensemble
# Created on: 10/12/23
# Version number: 10.12.23, making the ensemble classifier


# do the same as in part 2b
# but use the ensemble of the 3 models

# Importing libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Importing the dataset
data = pd.read_csv('data/breast-cancer.data', header=None)

del_rows = 0

# This reads the dataset from the file and adds a counter for the number of rows with "?" in them
with open('./output/part_2b_decision_trees_data_preprocessing.txt', 'r') as Preprocessed_Dataset:
    lines = Preprocessed_Dataset.readlines()
    for line in lines:
        if '?' in line:
            #Preprocessed_Dataset.write(line)
            del_rows += 1

Del_rows = str(del_rows)

df = pd.DataFrame(data)
Data_Length = data.shape[0]
for i in range(Data_Length - del_rows):
    if '?' in data.iloc[i].values:
        df.drop([i], axis=0, inplace=True)

# categorical data
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Encoding categorical data
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
for i in range(0, X.shape[1]):
    X[:, i] = label_encoder.fit_transform(X[:, i])


# endcode df 
for i in range(len(lines[0].split())):
    # Transform the categorical features in X_train into numerical features
    df.iloc[:, i] = label_encoder.fit_transform(df.iloc[:, i])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=97)

DT1 = DecisionTreeClassifier(criterion='entropy', max_depth = 1 , max_features = 10 ,random_state=97)
DT2 = DecisionTreeClassifier(criterion='entropy', max_depth = 3 , max_features = 10 ,random_state=97)
DT3 = DecisionTreeClassifier(criterion='entropy', max_depth= 10 , max_features = 10 ,random_state=97)

# train all the classifiers on the same training set   
DT1.fit(X_train, y_train)
DT2.fit(X_train, y_train)
DT3.fit(X_train, y_train)

# predict the test set for all the classifiers seperately
y_pred1 = DT1.predict(X_test)
y_pred2 = DT2.predict(X_test)
y_pred3 = DT3.predict(X_test)

# Produce a single prediction by majority vote
y_pred = np.array([])
for i in range(len(y_pred1)):
    if y_pred1[i] == y_pred2[i]:
        y_pred = np.append(y_pred, y_pred1[i])
    elif y_pred1[i] == y_pred3[i]:
        y_pred = np.append(y_pred, y_pred1[i])
    elif y_pred2[i] == y_pred3[i]:
        y_pred = np.append(y_pred, y_pred2[i])
    else:
        y_pred = np.append(y_pred, y_pred1[i])




# print the accuracy of the ensemble to a file
with open('./output/part_2c_ensemble_out.csv', 'w') as Ensemble:
    # make the header for the file
    # Ensemble.write('case_id,actual_class,DT1_prediction,DT2_prediction,DT3_prediction,Ensemble_prediction\n')
    for i in range(len(y_pred)):

        # case id
        Ensemble.write(str(i+1) + ',')

        # actual class label 
        Ensemble.write(str(y_test[i]) + ',')

        #DT1 prediction
        Ensemble.write(str(y_pred1[i]) + ',')

        #DT2 prediction 
        Ensemble.write(str(y_pred2[i]) + ',')

        #DT3 prediction
        Ensemble.write(str(y_pred3[i]) + ',')

        #Ensemble prediction
        Ensemble.write(str(y_pred[i]))
        Ensemble.write('\n')    

# finding the accuracy of the ensemble
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the ensemble is: ', accuracy)

# print the accuracy of the ensemble to a file
with open('./output/part_2c_ensemble_out.csv', 'a') as Ensemble:
    Ensemble.write('Accuracy of the ensemble is: ' + str(accuracy) + '\n')