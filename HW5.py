#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:06:51 2018

@author: yizhouwang
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA


# read dataset
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/'
                      'wine.data',header = None)
df_wine.columns = ['Class label', 'Alcohol',
                  'Malic acid', 'Ash',
                  'Alcalinity of ash',
                  'Magnesium',
                  'Total phenols',
                  'Flavanoids',
                  'Nonflavanoid ohenols',
                  'Proanthocyanins',
                  'Color intensity','Hue',
                  'OD280/OD315 of diluted wines',
                  'Proine']
df_wine.describe()
df_wine.head()

######EDA
sns.pairplot(df_wine, size=2.5)
plt.tight_layout()
plt.savefig('scatter_diagram.png', dpi=300)
plt.show()

cm = np.corrcoef(df_wine.values.T)
sns.set(font_scale=1.2)
mask = np.zeros_like(cm)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 6}, xticklabels=df_wine.columns.values, yticklabels=df_wine.columns.values)

plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(20, 20)
fig.savefig('heatmap.png', dpi=100)
plt.show()


#####Split dataset
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

###logistic regression baseline
lr = LogisticRegression(C=1.0, random_state=1)
lr.fit(X_train_std, y_train)
train_pred_lr = lr.predict(X_train_std)
test_pred_lr = lr.predict(X_test_std)
print ('Baseline: lr')
print ('trainset: ', accuracy_score(y_train, train_pred_lr))
print ('testset: ', accuracy_score(y_test, test_pred_lr))

# Part 2: SVM baseline
svm = SVC(kernel='linear', random_state=1, C=1.0)
svm.fit(X_train_std, y_train)
train_pred_svm = svm.predict(X_train_std)
test_pred_svm = svm.predict(X_test_std)
print ('Baseline: svm')
print ('trainset: ', accuracy_score(y_train, train_pred_svm))
print ('testset: ', accuracy_score(y_test, test_pred_svm))

### PCA: lr
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
train_pred_lr_pca = lr.predict(X_train_pca)
test_pred_lr_pca = lr.predict(X_test_pca)
print ('PCA: lr')
print ('trainset: ', accuracy_score(y_train, train_pred_lr_pca))
print ('testset: ', accuracy_score(y_test, test_pred_lr_pca))

### PCA: SVM
svm_pca = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_pca, y_train)
train_pred_svm_pca = svm.predict(X_train_pca)
test_pred_svm_pca = svm.predict(X_test_pca)
print ('PCA: svm')
print ('trainset: ', accuracy_score(y_train, train_pred_svm_pca))
print ('testset: ', accuracy_score(y_test, test_pred_svm_pca))


### LDA: lr
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr.fit(X_train_lda, y_train)
train_pred_lr_lda = lr.predict(X_train_lda)
test_pred_lr_lda = lr.predict(X_test_lda)
print ('LDA: lr')
print ('trainset: ', accuracy_score(y_train, train_pred_lr_lda))
print ('testset: ', accuracy_score(y_test, test_pred_lr_lda))

### LDA: SVM
svm = SVC(kernel='linear', random_state=1, C=1.0)
svm.fit(X_train_lda, y_train)
train_pred_svm_lda = svm.predict(X_train_lda)
test_pred_svm_lda = svm.predict(X_test_lda)
print ('LDA: svm')
print ('trainset: ', accuracy_score(y_train, train_pred_svm_lda))
print ('testset: ', accuracy_score(y_test, test_pred_svm_lda))

### Kernel: lr
kpca = KernelPCA(n_components=2, 
              kernel='rbf', gamma=0.5)
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.fit_transform(X_test_std, y_test)
lr.fit(X_train_kpca, y_train)
train_pred_lr_kpca = lr.predict(X_train_kpca)
test_pred_lr_kpca= lr.predict(X_test_kpca)
print ('kpca: lr')
print ('trainset: ', accuracy_score(y_train, train_pred_lr_kpca))
print ('testset: ', accuracy_score(y_test, test_pred_lr_kpca))

### Kernel: SVM
svm_kpca = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_kpca, y_train)
train_pred_svm_kpca = svm.predict(X_train_kpca)
test_pred_svm_kpca = svm.predict(X_test_kpca)
print ('kpca: svm')
print ('trainset: ', accuracy_score(y_train, train_pred_svm_kpca))
print ('testset: ', accuracy_score(y_test, test_pred_svm_kpca))

### different gamma
for i in range(1,7):
    kpca = KernelPCA(n_components=2, 
              kernel='rbf', gamma=i)
    X_train_kpca = kpca.fit_transform(X_train_std, y_train)
    X_test_kpca = kpca.fit_transform(X_test_std, y_test)
    lr.fit(X_train_kpca, y_train)
    train_pred_lr_kpca = lr.predict(X_train_kpca)
    test_pred_lr_kpca= lr.predict(X_test_kpca)
    print('***************************************************')
    print ('Gamma: ', i)
    print ('kpca: lr')
    print ('trainset: ', accuracy_score(y_train, train_pred_lr_kpca))
    print ('testset: ', accuracy_score(y_test, test_pred_lr_kpca))

    svm_kpca = SVC(kernel='linear', C=1.0, random_state=1)
    train_pred_svm_kpca = svm.predict(X_train_kpca)
    test_pred_svm_kpca = svm.predict(X_test_kpca)
    print ('kpca: svm')
    print ('trainset: ', accuracy_score(y_train, train_pred_svm_kpca))
    print ('testset: ', accuracy_score(y_test, test_pred_svm_kpca))


print("My name is YizhouWang")
print("My NetID is: yizhouw4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")