# -*- coding: utf-8 -*-
"""
Created on Wed May 18 20:46:04 2022

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

url = 'diabetes.csv'
data = pd.read_csv(url)

# Tratamiento data

data.Age.replace(np.nan, 33, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
data.dropna(axis=0,how='any', inplace=True)

data.drop(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'], axis=1, inplace=True)

# Dividir la data en dos

data_train = data[:385]
data_test = data[385:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)

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