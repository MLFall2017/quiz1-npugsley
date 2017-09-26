# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:01:06 2017

@author: Noemi Pugsley de Souza
"""
import numpy as np
from numpy import linalg as LA
import csv as csv
from csv import reader
from scipy import stats
import matplotlib.pyplot as plt
 

# Exercise 1

#Load Data 
data = csv.reader(open('C:\Users\pugsleno\Desktop\Pessoal Docs\UNC\MachineLearning\dataset_1.csv'))

for row in data:
    print row

# Specify Variables
X = []
Y = []
Z = []


data.shape    
    
# Calculate the variance of every variable

np.var(data)    


# Exercise 3.2 - Part 3.1 Printed

a = np.array([[0,-1],[2,3]], int)
print a
np.linalg.det(a)      # finds the determinant of matrix a
print np.linalg.det(a)

# eigenvalues and eigenvetors of a matrix
vals, vecs = np.linalg.eig(a)
print vals
print vecs



