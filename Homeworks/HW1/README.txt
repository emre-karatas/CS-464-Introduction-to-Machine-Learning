README
======

Author: EMRE KARATAŞ
ID: 22001641
Course: CS 464 
Homework Assignment: 01

Description:
------------
This README file provides instructions on how to execute the 'q3main.py' script for Homework 01, which includes the implementation of different Naive Bayes classifiers for a given dataset.

Prerequisites:
--------------
Ensure you have Python installed on your system. The script is compatible with Python 3.8 (Conda) versions.

Required Python packages:
- pandas
- numpy
- matplotlib

Installation of packages using pip:

pip install pandas numpy matplotlib


Running the Program:
--------------------
To execute the 'q3main.py' script, use the terminal (or command prompt) to navigate to the directory containing the script and the dataset files. 

IT IS IMPORTANT TO KEEP DATA FILES WITHIN THE FILE CONTAINING "q3main.py".

Run the following command:

python q3main.py


Menu Options:
-------------
Upon execution, the script will display a menu with the following options:
1. Print class counts for training and test sets
2. Plot class distribution
3. Calculate and display prior probabilities
4. Calculate and display word occurrences in 'Tech' class
5. Train a Multinomial Naive Bayes Model
6. Train a Multinomial Naive Bayes Model with Smoothing
7. Train a Bernoulli Naive Bayes Model
8. Exit

Users can input the number corresponding to their choice to execute a particular section of the homework.

Default Values:
---------------
The script uses a default alpha value of 1 for additive smoothing in the Naive Bayes classifiers.

Outputs:
--------
The outputs will be displayed in the terminal. They include:
- Class counts for the training and test sets
- Pie charts showing the class distribution
- Prior probabilities for each class in the training set
- Word occurrences in the 'Tech' category along with their log probabilities
- Accuracy and confusion matrix for the Multinomial and Bernoulli Naive Bayes models

Please refer to the in-script comments for additional details on the outputs and their interpretation.


