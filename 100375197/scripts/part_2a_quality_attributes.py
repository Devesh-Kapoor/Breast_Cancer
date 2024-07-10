# Description: Making functions to assess quality of attributes unsing information gain and gini index and chi square
# Created on: 03/12/2023
# Version number: 03.12.23, Creating file and functions

# Importing libraries
import numpy as np

# Assume a 2x2 contingency table that surmises 
# the relationship between two cases across different attribute values
# 1. Calculate the information gain
# 2. Calculate the gini impurity
# 3. Calculate the chi square

# Create a 2x2 contingency table
# The layout for the contingency table is as follows
# [Positive/Yes,Negative/Yes][Positive/No,Negative/No]



# Function to calculate information gain
# This function takes in a contingency table and calculates the information gain by
# calclating the entropy of the root node and the entropy of the positive and negative values
# and then calculating the information gain using the formula
# when the information gain is calculated, it is printed out
# and the information gain is returned
# if there are undefined values, the information gain is set to NaN
def get_information_gain(your_contingency_table):
    # Get total amount of elements in the contingency table
    total = np.sum(your_contingency_table)

    # Getting the values from the contingency table
    positive_yes = your_contingency_table[0][0]
    negative_yes = your_contingency_table[0][1]
    positive_no = your_contingency_table[1][0]
    negative_no = your_contingency_table[1][1]

    # Calculate the total amount of positive and negative values
    total_positive = positive_yes + positive_no
    total_negative = negative_yes + negative_no

    # Calculate the total entropy
    root_entropy = -((total_positive/total)*np.log2(total_positive/total)) - ((total_negative/total)*np.log2(total_negative/total))

    # Calculating the entropy for the positive and negative values
    entropy_positive = -((positive_yes/(positive_yes+negative_yes))*np.log2(positive_yes/(positive_yes+negative_yes))) - ((negative_yes/(positive_yes+negative_yes))*np.log2(negative_yes/(positive_yes+negative_yes)))
    entropy_negative = -((positive_no/(positive_no+negative_no))*np.log2(positive_no/(positive_no+negative_no))) - ((negative_no/(positive_no+negative_no))*np.log2(negative_no/(positive_no+negative_no)))

    #Calculate the entropy for the positive and negative values
    positive_entropy = (positive_yes+negative_yes)/total*entropy_positive
    negative_entropy = (positive_no+negative_no)/total*entropy_negative

    # Calculate the information gain
    information_gain = root_entropy - positive_entropy - negative_entropy

    # Print all values
    # print(positive_yes)
    # print(negative_yes)
    # print(positive_no)
    # print(negative_no)
    # print(root_entropy)
    # print(entropy_positive)
    # print(entropy_negative)

    # Print the information gain
    print("Information Gain:",information_gain)
    
    return information_gain

# Function to calculate gini impurity
# This function takes in a contingency table and calculates the gini impurity by
# calclating the gini impurity of the positive and negative values
# and then calculating the weighted gini impurity using the formula
# when the gini impurity is calculated, it is printed out
# and the gini impurity is returned
# if there are undefined values, the gini impurity is set to NaN
def get_gini_impurity(your_contingency_table):

    #Get total amount of elements in the contingency table
    total = np.sum(your_contingency_table)

    # Getting the values from the contingency table
    positive_yes = your_contingency_table[0][0]
    negative_yes = your_contingency_table[0][1]
    positive_no = your_contingency_table[1][0]
    negative_no = your_contingency_table[1][1]

    # Calculating the gini impurity for the positive and negative values
    gini_impurity_positive = 1 - ((positive_yes/(positive_yes+negative_yes))**2) - ((negative_yes/(positive_yes+negative_yes))**2)
    gini_impurity_negative = 1 - ((positive_no/(positive_no+negative_no))**2) - ((negative_no/(positive_no+negative_no))**2)
    
    #calculate the weighted gini impurity
    gini_impurity = ((positive_yes+negative_yes)/total)*gini_impurity_positive + ((positive_no+negative_no)/total)*gini_impurity_negative
    
    # Printing the values
    # print(positive_yes)
    # print(negative_yes)
    # print(positive_no)
    # print(negative_no)
    # print(gini_impurity_positive)
    # print(gini_impurity_negative)

    # Print the gini impurity
    print("Gini impurity:",gini_impurity)

    return gini_impurity

# Function to calculate chi square
# This function takes in a contingency table and calculates the chi square by
# calclating the expected values and then calculating the chi square using the formula
# when the chi square is calculated, it is printed out
# and the chi square is returned
# if there are undefined values, the chi square is set to NaN
def get_chi2(your_contingency_table):

    # Calculate the total amount of elements in the contingency table
    total = np.sum(your_contingency_table)

    # Calculate the row and column totals
    row_totals = np.sum(your_contingency_table, axis=1)
    column_totals = np.sum(your_contingency_table, axis=0)

    # Calculate the expected values
    expected = np.outer(row_totals, column_totals) / total

    # Calculate the chi2 statistic
    chi2 = np.sum((your_contingency_table - expected)**2 / expected)

    # Print all values
    # print(row_totals)
    # print(column_totals)
    # print(total)
    # print(expected)

    # Print the chi2 statistic
    print("Chi-squared:",chi2)
    return chi2


# Creating the contingency tables for the attributes

# Attribute 1: Stiff neck
contingency_table_stiff_neck = np.array([[4,1],[1,4]])
# Attribute 2: Headache
contingency_table_headache = np.array([[3,1],[2,4]])
# Attribute 3: Spots
contingency_table_spots = np.array([[4,2],[1,3]])

############################################################################################################
# Make a file to put the results in
f = open("./output/part_2a_quality_attributes_out.txt", "w")
############################################################################################################
# Values for stiff neck

# Printing and writing the attribute name into a file
print("Attribute name: Stiff neck")
# Put into a file
f.write("Attribute name: Stiff neck")


# Print and writing to a file contents of contingency table
print("Contingency table:",contingency_table_stiff_neck)
f.write("\nContingency table: " + str(contingency_table_stiff_neck))

# Output and write get_information_gain
f.write("\nInformation Gain: " + str(get_information_gain(contingency_table_stiff_neck)))

# Output and write get_gini_impurity
f.write("\nGini impurity: " + str(get_gini_impurity(contingency_table_stiff_neck)))

# Output and write get_chi2
f.write("\nChi-squared: " + str(get_chi2(contingency_table_stiff_neck)))

f.write("\n")

print("\n")
############################################################################################################         
# Values for headache

# Printing and writing the attribute name
print("Attribute name: Headache")
f.write("\nAttribute name: Headache")

# Printing to console and writing to a file the contents of contingency table
print("Contingency table:",contingency_table_headache)
f.write("\nContingency table: " + str(contingency_table_headache))

# Outputing and writing get_information_gain
f.write("\nInformation Gain: " + str(get_information_gain(contingency_table_headache)))

# Outputing and writing get_gini_impurity
f.write("\nGini impurity: " + str(get_gini_impurity(contingency_table_headache)))

# Outputing and writing get_chi2
f.write("\nChi-squared: " + str(get_chi2(contingency_table_headache)))

print("\n")
f.write("\n")

############################################################################################################

# Values for spots

# Printing and writing the attribute name
print("Attribute name: Spots")
f.write("\nAttribute name: Spots")

# Print to console  and writing to a file the contents of contingency table
print("Contingency table:",contingency_table_spots)
f.write("\nContingency table: " + str(contingency_table_spots))

# Outputing and writing get_information_gain
f.write("\nInformation Gain: " + str(get_information_gain(contingency_table_spots)))

# Outputing and writing get_gini_impurity
f.write("\nGini impurity: " + str(get_gini_impurity(contingency_table_spots)))

# Outputing and writing get_chi2
f.write("\nChi-squared: " + str(get_chi2(contingency_table_spots)))

print("\n")
f.write("\n")



