# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:09:41 2022

@author: Connor
"""
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Below is how we prepare data for modeling 
cancer = load_breast_cancer()
X=cancer.data 
y=cancer.target
#scaling data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Now we need to insert PCA to transform data
pca = PCA(n_components=2)
X_trained = pca.fit_transform(X_scaled)
print('Scaled d-set shape:', X_scaled.shape)
print('PCA transformed dataset shape:', X_trained.shape)
print('PCA component shape:', pca.components_.shape)
# Plotting the first two principal components/features
fig, ax = plt.subplots(figsize=(7,7), dpi=100)
ax.scatter(X_trained[:, 0], X_trained[:, 1], c=y, edgecolor='none', alpha=0.75, cmap=plt.cm.get_cmap('nipy_spectral', 10))
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.set_title('First 2 components of transformed dataset')
# 3D plot with the first 3 features of cancer data set

fig2 = plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax2 = Axes3D(fig2, rect=[0, 0, .95, 1], elev=10, azim=10)
ax2.scatter(X_scaled[:,0],X_scaled[:,1],X_scaled[:,2], c=y, cmap=cmap)
ax2.set_xlabel('First principal component')
ax2.set_ylabel('Second principal component')
ax2.set_zlabel('Third principal component')
ax2.set_title('First 3 components of transformed dataset')
#3D plot with 2 of the first features that are contained in the transformed cancer.data
fig3 = plt.figure(figsize=(10, 8))
ax3 = Axes3D(fig3, rect=[0, 0, .95, 1], elev=10, azim=10)
ax3.scatter(X_trained[:,0],X_trained[:,1], c=y, cmap=cmap)
ax3.set_xlabel('First principal component')
ax3.set_ylabel('Second principal component')
ax3.set_title('First 2 components of transformed data')

