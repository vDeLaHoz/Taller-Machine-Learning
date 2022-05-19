# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:25:19 2022

@author: vdelahoz6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Obtener data

url = 'bank-full.csv'
data = pd.read_csv(url)

# Tratamiento data

data.age.replace(np.nan, 41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0,how='any', inplace=True)
data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)
data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace=True)
data.default.replace(['yes', 'no'], [0, 1], inplace=True)
data.housing.replace(['yes', 'no'], [0, 1], inplace=True)
data.loan.replace(['yes', 'no'], [0, 1], inplace=True)
data.contact.replace(['cellular', 'unknown', 'telephone'], [0, 1, 2], inplace=True)
data.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace=True)
data.y.replace(['yes', 'no'], [0, 1], inplace=True)

data.drop(['balance', 'duration', 'campaign', 'pdays', 'previous', 'job', 'day', 'month'], axis=1, inplace=True)

# Dividir la data en dos

data_train = data[:22607]
data_test = data[22607:]

x = np.array(data_train.drop(['housing'], 1))
y = np.array(data_train.housing)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['housing'], 1))
y_test_out = np.array(data_test.housing)

#Modelos:

# Regresión Logística

# Seleccion del modelo
rl = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
rl.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'Accuracy de Entrenamiento de Entrenamiento: {rl.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'Accuracy de Test de Entrenamiento: {rl.score(x_test, y_test)}')

# Accuracy de Validación
print(f'Accuracy de Validación: {rl.score(x_test_out, y_test_out)}')

# Validacion cruzada modelo Regresión Logística

kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
rl= LogisticRegression(solver='lbfgs', max_iter=7600)

for train, test in kfold.split(x, y):
    rl.fit(x[train], y[train])
    scores_train_train = rl.score(x[train], y[train])
    scores_test_train = rl.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = rl.predict(x_test_out)
    

print('*'*50)
print('Regresion Logistica Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {rl.score(x_test_out, y_test_out)}')

# Matriz de confusión
print('*'*50)
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusión")

# Metricas
precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'y real: {y_test_out}')
print(f'y predicho: {y_pred}')


# MAQUINA DE SOPORTE VECTORIAL

# Selecciona del modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'Accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'Accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'Accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# Validacion cruzada modelo Maquina de soporte vectorial

kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
svc= SVC(gamma='auto')

for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)
    

print('*'*50)
print('Maquina de soporte vectorial Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# Matriz de confusión
print('*'*50)
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusión")

# Metricas
precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'y real: {y_test_out}')
print(f'y predicho: {y_pred}')



# ARBOL DE DECISIÓN

# Seleccion del modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'Accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'Accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'Accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

# Validacion cruzada modelo Arbol de Decision

kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
arbol= DecisionTreeClassifier()

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)
    

print('*'*50)
print('Arbol De Decision Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

# Matriz de confusión
print('*'*50)
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusión")

# Metricas
precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'y real: {y_test_out}')
print(f'y predicho: {y_pred}')


# DecisionTreeRegressor

# Seleccion del modelo
treeR = DecisionTreeRegressor()

# Entreno el modelo
treeR.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('DecisionTreeRegressor')

# Accuracy de Entrenamiento de Entrenamiento
print(f'Accuracy de Entrenamiento de Entrenamiento: {treeR.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'Accuracy de Test de Entrenamiento: {treeR.score(x_test, y_test)}')

# Accuracy de Validación
print(f'Accuracy de Validación: {treeR.score(x_test_out, y_test_out)}')

# Validacion cruzada modelo DecisionTreeRegressor

kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
treeR = DecisionTreeRegressor()

for train, test in kfold.split(x, y):
    treeR.fit(x[train], y[train])
    scores_train_train = treeR.score(x[train], y[train])
    scores_test_train = treeR.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = treeR.predict(x_test_out)
    

print('*'*50)
print('DecisionTreeRegressor Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {treeR.score(x_test_out, y_test_out)}')

# Matriz de confusión
print('*'*50)
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusión")

# Metricas
precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'y real: {y_test_out}')
print(f'y predicho: {y_pred}')



#RANDOM FOREST

# Seleccion del modelo
rf = RandomForestClassifier()

# Entreno el modelo
rf.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Random Forest')

# Accuracy de Entrenamiento de Entrenamiento
print(f'Accuracy de Entrenamiento de Entrenamiento: {rf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'Accuracy de Test de Entrenamiento: {rf.score(x_test, y_test)}')

# Accuracy de Validación
print(f'Accuracy de Validación: {rf.score(x_test_out, y_test_out)}')

# Validacion cruzada RANDOM FOREST

kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
rf = RandomForestClassifier()

for train, test in kfold.split(x, y):
    rf.fit(x[train], y[train])
    scores_train_train = rf.score(x[train], y[train])
    scores_test_train = rf.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = rf.predict(x_test_out)
    

print('*'*50)
print('Random Forest Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')

# Matriz de confusión
print('*'*50)
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusión")

# Metricas
precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'y real: {y_test_out}')
print(f'y predicho: {y_pred}')