# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:25:19 2022

@author: vdelahoz6
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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