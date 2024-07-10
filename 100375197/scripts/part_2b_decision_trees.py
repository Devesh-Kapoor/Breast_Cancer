# Description: Implementation of Decision Trees classifiers
# Created on: 04/12/2023
# Version number: 09.12.23, implement decision trees classifiers

# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

# Part 1 

#Load the UCI Breast Cancer (categorical) dataset into X and y variables
#following the sklearn convention to load the feature data into a variable named
#X and the target/class column to variable named y. Read the dataset
#documentation to learn which is column containing the target/class label.

# Load the dataset breast-cancer.data into X and y variables
# The dataset is loaded from the data folder
data = pd.read_csv('./data/breast-cancer.data', header=None)
# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values

# Printing the values of X and y
#print('X:', X)
#print('y:', y)


# create a file named part_2b_decision_trees_data_preprocessing.data
Preprocessed_Dataset = open('./output/part_2b_decision_trees_data_preprocessing.txt', 'w')

# Part 2 
# Getting rid of data that could harm the overall accuracy of the model, such as missing data denoted by
# a question mark '?'.

# Keeping track of deleted rows 
del_rows = 0

# This writes the dataset into the file
with open('./output/part_2b_decision_trees_data_preprocessing.txt', 'w')as Preprocessed_Dataset:
    Preprocessed_Dataset.write(data.to_string())

# This reads the dataset from the file and adds a counter for the number of rows with "?" in them
with open('./output/part_2b_decision_trees_data_preprocessing.txt', 'r') as Preprocessed_Dataset:
    lines = Preprocessed_Dataset.readlines()
    for line in lines:
        if '?' in line:
            #Preprocessed_Dataset.write(line)
            del_rows += 1
            
Del_rows = str(del_rows)

# This loads all the data into the file 
with open('./output/part_2b_decision_trees_data_preprocessing.txt', 'w') as Preprocessed_Dataset:
    Preprocessed_Dataset.write("Dataset: Breast Cancer Wisconsin Data Set\n")
    Preprocessed_Dataset.write("Number of cases removed:" + Del_rows + "\n")
    Preprocessed_Dataset.write("Cases:" + str(len(lines) - del_rows) + "\n")
    Preprocessed_Dataset.write("Attributes:" + str(len(lines[0].split()) - 1) + "\n")

# This part reads the data to put it into a dataframe, and then removes the rows with missing data
# and then writes the dataframe to a file which does not contain missing data, so it can be used to 

df = pd.DataFrame(data)
# print dataset by line as an array
Data_Length = data.shape[0]
for i in range(Data_Length - del_rows):
    if '?' in data.iloc[i].values:
        df.drop([i], axis=0, inplace=True)


# load the df into X and y
X = df.iloc[:, 1:].values   
y = df.iloc[:, 0].values

# Part 3 

# Transform categorical features in a dataset into numerical features using sklearn 
# preprocessing.LabelEncoder. Print the transformed dataset.

# Create a LabelEncoder object
Label_Encoder = LabelEncoder()

# endcode df 
for i in range(len(lines[0].split())):
    # Transform the categorical features in X_train into numerical features
    df.iloc[:, i] = Label_Encoder.fit_transform(df.iloc[:, i])

# Part 4

# Update part_2b_decision_trees_data_preprocessing.txt with the following information:
# original column name: number of binary columns generated for the categorical feature

# The names of the columns are:
# Breast cancer features 
# 1. Class: no-recurrence-events, recurrence-events
# 2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69,
#         70-79, 80-89, 90-99.
# 3. menopause: lt40, ge40, premeno.
# 4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29,
#                30-34, 35-39, 40-44, 45-49, 50-54, 55-59
# 5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17,
#               18-20, 21-23, 24-26, 27-29, 30-32, 33-35,
#               36-39
# 6. node-caps: yes, no.
# 7. deg-malig: 1, 2, 3.
# 8. breast: left, right.
# 9. breast-quad: left-up, left-low, right-up,	right-low, central.
# 10. irradiat:	yes, no.


# This is for making a list of all the categories in each column as a numeric value
Class = []
age = []
menopause = []
tumor_size = []
inv_nodes = []
node_caps = []
deg_malig = []
breast = []
breast_quad = []
irradiat = []

# This runs through all the values adding them to the list, if the number is already in the list
# it will not add it again
for i in range(df.shape[0]):
    for j in range(len(lines[0].split())):
        if j == 0:
            if df.iloc[i].values[j] not in Class:
                Class.append(df.iloc[i].values[j])
        elif j == 1:
            if df.iloc[i].values[j] not in age:
                age.append(df.iloc[i].values[j])
        elif j == 2:
            if df.iloc[i].values[j] not in menopause:
                menopause.append(df.iloc[i].values[j])
        elif j == 3:
            if df.iloc[i].values[j] not in tumor_size:
                tumor_size.append(df.iloc[i].values[j])
        elif j == 4:
            if df.iloc[i].values[j] not in inv_nodes:
                inv_nodes.append(df.iloc[i].values[j])
        elif j == 5:
            if df.iloc[i].values[j] not in node_caps:
                node_caps.append(df.iloc[i].values[j])
        elif j == 6:
            if df.iloc[i].values[j] not in deg_malig:
                deg_malig.append(df.iloc[i].values[j])
        elif j == 7:
            if df.iloc[i].values[j] not in breast:
                breast.append(df.iloc[i].values[j])
        elif j == 8:
            if df.iloc[i].values[j] not in breast_quad:
                breast_quad.append(df.iloc[i].values[j])
        elif j == 9:
            if df.iloc[i].values[j] not in irradiat:
                irradiat.append(df.iloc[i].values[j])


# Add this data into the part_2b_decision_trees_data_preprocessing.txt file
Appending_categories = open('./output/part_2b_decision_trees_data_preprocessing.txt', 'a')
Appending_categories.write("\n")
Appending_categories.write("Class: " + str(len(Class)) + "\n")
Appending_categories.write("age: " + str(len(age)) + "\n")
Appending_categories.write("menopause: " + str(len(menopause)) + "\n")
Appending_categories.write("tumor_size: " + str(len(tumor_size)) + "\n")
Appending_categories.write("inv_nodes: " + str(len(inv_nodes)) + "\n")
Appending_categories.write("node_caps: " + str(len(node_caps)) + "\n")
Appending_categories.write("deg_malig: " + str(len(deg_malig)) + "\n")
Appending_categories.write("breast: " + str(len(breast)) + "\n")
Appending_categories.write("breast_quad: " + str(len(breast_quad)) + "\n")
Appending_categories.write("irradiat: " + str(len(irradiat)) + "\n")

Appending_categories.close()

# Part 5

# y is currently type object, I need to change it to type int
y=y.astype('int')

# Split the dataset into training and test set using sklearn.model_selection.train_test_split.
# Manually Split the dataset into training and test set
# Use 80% of the data for training and 20% for testing
# Use random_state=97 to get same random split everytime

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2,random_state=97)

# Print the shape of X_train, X_test, y_train, y_test
# print('Shape of X_train:', X_train.shape)
# print('Shape of X_test:', X_test.shape)
# print('Shape of y_train:', y_train.shape)
# print('Shape of y_test:', y_test.shape)


for i in range(len(lines[0].split()) - 1):
    # Transform the categorical features in X_train into numerical features
    X_train[:, i] = Label_Encoder.fit_transform(X_train[:, i])
    # Transform the categorical features in X_test into numerical features
    X_test[:, i] = Label_Encoder.fit_transform(X_test[:, i])

# Make a decision tree classifier, as a class and modify configuration parameters such as the 
# quality measure and tree stopping criteria. My Decision tree will have criterion='entropy' and
# max_depth=2.

# Create a DecisionTreeClassifier object
Decision_Tree_Classifier = DecisionTreeClassifier(random_state=97, criterion='entropy', max_depth=2)

# Train the DecisionTreeClassifier model

Decision_Tree_Classifier.fit(X_train, y_train)

# train classifier using training set and compute accuracy as well as balance scores on test set
# print to console and write into a plain file called part_2b_decision_trees_out.txt

# This is the accuracy of the model
Accuracy = accuracy_score(y_test, Decision_Tree_Classifier.predict(X_test))

# This is the balanced accuracy of the model
Balanced_Accuracy = balanced_accuracy_score(y_test, Decision_Tree_Classifier.predict(X_test))

print('Accuracy:', Accuracy)
print('Balanced Accuracy:', Balanced_Accuracy)

# Making a decsiion tree classifier plot

# This is the name of the features
features = ['age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat']

# This is the name of the classes
classes = ['no-recurrence-events', 'recurrence-events']

# This will make a graph using the decision tree classifier and save it as a png file
fig = plt.figure(figsize=(9,9))
Graph = tree.plot_tree(Decision_Tree_Classifier, 
                   feature_names=features,  
                   class_names=classes)

# Saving the graph as a png file into the output folder
fig.savefig("./output/part_2b_decision_trees_tree_out.png")

# This will show the graph
plt.show()


# Making a text based decision tree classifier plot. Using this will allow me to see the decision tree
# in a text based format

# tree_text = export_text(Decision_Tree_Classifier,

#                         feature_names = list(features))

# # prints to console
# print(tree_text)

print("Decision Tree Classifier\n"+"Criterion: entropy\n"+"Max Depth: 2\n"+"Accuracy: "+ str(Accuracy) + "\n"+"Balanced Accuracy: "+ str(Balanced_Accuracy))

# Putting the decision tree classifier configuration parameters into the file
with open('./output/part_2b_decision_trees_out.txt', 'w') as Decision_Tree_Classifier_Configuration_Parameters:
    Decision_Tree_Classifier_Configuration_Parameters.write("Decision Tree Classifier\n")
    Decision_Tree_Classifier_Configuration_Parameters.write("Criterion: entropy\n")
    Decision_Tree_Classifier_Configuration_Parameters.write("Max Depth: 2\n")
    Decision_Tree_Classifier_Configuration_Parameters.write("Accuracy: " + str(Accuracy) + "\n")
    Decision_Tree_Classifier_Configuration_Parameters.write("Balanced Accuracy: " + str(Balanced_Accuracy) + "\n")


# Making more decision tree classifiers with different configurations for max_depth 

Decision_Tree_Classifier_Configuration_Parameters = open('./output/part_2b_decision_trees_max_depths_out.txt', 'w')

Acc_array = []
Balanced_Acc_array = []

fig2 = plt.figure(figsize=(10,10))
plt.title('Accuracy and Balanced Accuracy Scores for Each of the 10 Models')

for i in range(1,11):
    # Create a DecisionTreeClassifier object
    Decision_Tree_Classifier = DecisionTreeClassifier(random_state= 97,criterion='entropy', max_depth=i)

    # Train the DecisionTreeClassifier model

    Decision_Tree_Classifier.fit(X_train, y_train)

    # train classifier using training set and compute accuracy as well as balance scores on test set
    # print to console and write into a plain file called part_2b_decision_trees_out.txt

    # This is the accuracy of the model
    Accuracy = accuracy_score(y_test, Decision_Tree_Classifier.predict(X_test))

    # This is the balanced accuracy of the model
    Balanced_Accuracy = balanced_accuracy_score(y_test, Decision_Tree_Classifier.predict(X_test))

    Acc_array.append(Accuracy)
    Balanced_Acc_array.append(Balanced_Accuracy)

    print("Model id "+ str(i)+", max depth: "+str(i)+", accuracy: "+ str(Accuracy) +", balanced accuracy: "+str(Balanced_Accuracy))

    # Putting the decision tree classifier configuration parameters into the file
    with open('./output/part_2b_decision_trees_max_depths_out.txt', 'a') as Decision_Tree_Classifier_Configuration_Parameters:
        Decision_Tree_Classifier_Configuration_Parameters.write("Model id "+ str(i)+", max depth: "+str(i)+", accuracy: "+ str(Accuracy) +", balanced accuracy: "+str(Balanced_Accuracy)+" \n")

    

# Produce a plot showing the accuracy and balanced accuracy scores for each of the
# 10 models. Indicate on the plot the model configuration that achieved the highest
# accuracy and the model with the highest balanced accuracy. Save the plot as
# part_2b_decision_trees_max_depths_plot.png

# Making a compound bar chart to show the accuracy and balanced accuracy scores for each of the 10 models
br1 = np.arange(len(Acc_array))
br2 = [x + 0.25 for x in br1]

# providing the features for the bar chart
plt.bar(br1, Acc_array, color='r', width=0.25, label='Accuracy')
plt.bar(br2, Balanced_Acc_array, color='b', width=0.25, label='Balanced Accuracy')
# Since the numbers are quite large, I had to only show one of the highest for each of the bars
# This will show the highest accuracy and balanced accuracy on the graph
# The highest accuracy is shown as a gold bar and the highest balanced accuracy is shown as a silver bar
# since there are multiple highest accuracy scores, they will be gold to show that that they are the same
plt.bar(0,0,color = 'gold',label='Highest Accuracy = gold bar')
plt.bar(0,0,color = 'Silver',label='Highest Balance Accuracy = Silver bar')
plt.xlabel('Model ID')
plt.ylabel('Accuracy and Balanced Accuracy Scores')
plt.title('Accuracy and Balanced Accuracy Scores for Each of the 10 Models')
plt.legend(loc='lower right')
plt.xticks([r + 0.25 for r in range(len(Acc_array))], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

# iterating through the bars to find the highest accuracy and balanced accuracy
high_acc_done = 0
for i,value in enumerate(Acc_array):
    high_accuracy = max(Acc_array)

    # comparing the values to find the highest accuracy, so it can be shown on the graph
    if value == high_accuracy:
        if high_acc_done != 1:
            # puts text on the bar chart to show the highest accuracy
            plt.text(i, value, str(value), ha='center')
            high_acc_done =  1
        # change the color of the bar to green
        plt.bar(i, value, color='gold', width=0.25, label='Accuracy')

high_balance_done = 0   
for i,value in enumerate(Balanced_Acc_array):
    high_balanced_accuracy = max(Balanced_Acc_array)
    
    # comparing the values to find the highest balanced accuracy, so it can be shown on the graph
    if value == high_balanced_accuracy:
        if high_balance_done != 1:
            # puts text on the bar chart to show the highest balanced accuracy
            plt.text(i, value, str(value), ha='center')
            # change the color of the bar to green
            plt.bar(i+0.25, value, color='Silver', width=0.25, label='Balanced Accuracy')
            high_acc_done = 1

# This will show the graph
plt.show()


# This will save the plot as a png file
fig2.savefig("./output/part_2b_decision_trees_max_depths_plot.png")