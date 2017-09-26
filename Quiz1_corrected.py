# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:55:54 2017

@author: pugsleno
"""

##  Quiz #1 Correct

#______________________________________________________________________________
#        Load Libraries/Packages
#______________________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt           #plots framework
from numpy import linalg as LA
import pandas as pd
#from sklearn.decomposition import PCA
from matplotlib.mlab import PCA
#PCA.switch_backend('pgf') 


#Packages to support plot display in 3d

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


#______________________________________________________________________________
#  Calculate the variance of every variable in the data file.
#______________________________________________________________________________

# 1. Load Raw Data
    
in_file_name = "C:\Users\pugsleno\Desktop\Pessoal Docs\UNC\MachineLearning\Quiz#1\dataset_1.csv"
    
dataIn = pd.read_csv(in_file_name)                    # Read the Raw Data
    
# 2. Define Variables:
    
x = dataIn['x']
y = dataIn['y']
z = dataIn['z']

# Variance of x, y and z

variance_x = np.var(x)
variance_y = np.var(y)
variance_z = np.var(z)

print 'Variance X: ', variance_x 
print 'Variance Y: ', variance_y
print 'Variance Z: ', variance_z 

#______________________________________________________________________________
#  calculate the covariance between x and y, and between y and z
#______________________________________________________________________________

covariance_xy = np.cov(x,y, rowvar=False)  
covariance_yz = np.cov(y,z, rowvar=False)  

print 'Covariance XY: \n', covariance_xy 
print 'Covariance YZ: \n', covariance_yz

#______________________________________________________________________________
#   do PCA of all the data in the given data file using your own PCA module
#______________________________________________________________________________

# Step 1. Mean

mean_X = np.mean(x)
mean_Y = np.mean(y)
mean_Z = np.mean(z)

# Step 2. Mean Centered Data

std_X = x - mean_X
std_Y = y - mean_Y
std_Z = z - mean_Z


# Step 3. Covariance

covariance_xy = np.cov(x,y, rowvar=False)  
covariance_yz = np.cov(y,z, rowvar=False) 

# Step 4. Eigendecomposition of the covariance matrix

# Between XY

eigenValues_xy, eigenVectors_xy = np.linalg.eig(covariance_xy)
eigValSort= eigenValues_xy.argsort()[::-1]            
eigenValues_xy = eigenValues_xy[eigValSort]
eigenVectors_xy = eigenVectors_xy[:,eigValSort]


# Between YZ

eigenValues_yz, eigenVectors_yz = LA.eig(covariance_yz)
eigValSort= eigenValues_yz.argsort()[::-1]            
eigenValues_yz = eigenValues_yz[eigValSort]
eigenVectors_yz = eigenVectors_yz[:,eigValSort]



# Step 5.  PCA scores

# For X and Y

MeanCentered_xy = np.column_stack((std_X, std_Y))    #stacking X and Y std side by side on a matrix
pcaScores_xy = np.matmul(MeanCentered_xy, eigenVectors_xy)

# For Y and Z

MeanCentered_yz = np.column_stack((std_Y, std_Z))    #stacking X and Y std side by side on a matrix
pcaScores_yz = np.matmul(MeanCentered_yz, eigenVectors_yz)



# Step 6: Collect PCA results

# Between X and Y

RawData_xy = np.column_stack((x,y))                    #stacking X and Y std side by side on a matrix
pcaResults_xy = {'data': RawData_xy,
              'mean_centered_data': MeanCentered_xy,
              'PC_variance': eigenValues_xy,
              'loadings': eigenVectors_xy,
              'scores': pcaScores_xy}

# Between  Y and Z
RawData_yz = np.column_stack((y,z))                    #stacking X and Y std side by side on a matrix
pcaResults_yz = {'data': RawData_yz,
              'mean_centered_data': MeanCentered_yz,
              'PC_variance': eigenValues_yz,
              'loadings': eigenVectors_yz,
              'scores': pcaScores_yz}


print pcaResults_yz


VarianceExplained = 100 * pcaResults_xy['PC_variance'][0] / sum(pcaResults_xy['PC_variance'])
print "PC1 explains the Variance XY: " + str(round(VarianceExplained, 2,)) + '% variance\n'

VarianceExplained = 100 * pcaResults_yz['PC_variance'][1] / sum(pcaResults_yz['PC_variance'])
print "PC2 explains the Variance YZ: " + str(round(VarianceExplained, 2,)) + '% variance\n'


#____________________________________________________________________________
# 3.2 Use thelinalgmodule innumpyto find the eigenvalues and eigenvectors. 
# Are theythe same as your manual solution?
#____________________________________________________________________________


a = np.array([[0,-1],[2,3]], int)
print a
np.linalg.det(a)      # finds the determinant of matrix a
print np.linalg.det(a)

# eigenvalues and eigenvetors of a matrix
vals, vecs = np.linalg.eig(a)
print vals
print vecs




