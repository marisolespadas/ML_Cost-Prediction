#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# ## Lectura de datos

# ### Creación de los sets de datos de entrenamiento y prueba

# In[136]:


df = pd.read_csv('Ejercicios_entrevista/train_price.csv')
df.head()


# In[137]:


X = df.iloc[:,:-1].to_numpy()
y = df['price'].to_numpy()


# **Se construye el modelo usando el archivo "train_price", de donde se toma se forman sets de entrenamiento y validación.**
# Un 30% de los datos de "train_price" se toman para validación del modelo antes de pasar a la predicción de resultados.

# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
#x, y = np.array(x), np.array(y)


# Se leen y se forman sets de datos con "test_price", que se emplearán después de validar el modelo para la predicción de precios.

# In[131]:


df_toTest = pd.read_csv('Ejercicios_entrevista/test_price.csv')
df_toTest.head()


# In[132]:


X_toTest = df_test.drop("id" ,axis= 1).to_numpy()
X_toTest = np.array(X_toTest)


# ## Arboles de decisión
# 
# Se comienza con un modelo de árbol de decisión:

# In[117]:


#Iniciando algoritmo
dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

#Se ajusta al modelo el set de entrenamiento.
dtree.fit(X_train, y_train)


# Una vez que el modelo se ha construido, se realizan las predicciones con el fin de evaluar el modelo:

# In[118]:


#Predicción sobre el set de entrenamiento
pred_train_tree= dtree.predict(X_train)

#Metricas de evaluación (RMSE y R^2)
print(np.sqrt(mean_squared_error(y_train,pred_train_tree)))
print(r2_score(y_train, pred_train_tree))


# In[119]:


#Predicción sobre el set de prueba
pred_test_tree= dtree.predict(X_test)

#Metricas de evaluación (RMSE y R^2)
print(np.sqrt(mean_squared_error(y_test,pred_test_tree))) 
print(r2_score(y_test, pred_test_tree))


# Los resultados anteriores muestran que el RMSE es 5776.432 para los datos de entrenamiento y 5562.199 para los datos de prueba. Por otro lado, el valor R-cuadrado es 75.05 por ciento para los datos de entrenamiento y 77.5 por ciento para los datos de prueba. Lo cual es decentes pero se puede mejorar ajustando algunos parámetros.
# 
# Se cambia el valor de max_depth para observar como afecta el rendimiento del modelo:

# In[120]:


#Parámetro 'max_depth' cambia a 2 y 5, respectivamente.
dtree1 = DecisionTreeRegressor(max_depth=2)
dtree2 = DecisionTreeRegressor(max_depth=5)
dtree1.fit(X_train, y_train)
dtree2.fit(X_train, y_train)


# In[121]:


#Predicción sobre los datos de entrenamiento
tr1 = dtree1.predict(X_train)
tr2 = dtree2.predict(X_train)


# In[122]:


#Predicción sobre los datos de prueba
y1 = dtree1.predict(X_test)
y2 = dtree2.predict(X_test)


# **Métricas de evaluación (RMSE y R-squared) para el primer árbol de regresión ('dtree1').**

# In[123]:


#Metricas de evaluación (RMSE y R^2)
print(np.sqrt(mean_squared_error(y_train,tr1))) 
print(r2_score(y_train, tr1))


# In[124]:


#Metricas de evaluación (RMSE y R^2)
print(np.sqrt(mean_squared_error(y_test,y1))) 
print(r2_score(y_test, y1)) 


# Los resultados anteriores para el modelo 'dtree1' muestra que el RMSE es 5873.959 para los datos de entrenamiento y 5690.32 para los datos de prueba. El valor R-cuadrado es 74.21 por ciento para los datos de entrenamiento y 76.49 por ciento para los datos de prueba. Este modelo con respecto al anterior tiene un rendimiento inferior en ambas métricas de evaluación.

# **Métricas de evaluación (RMSE y R-squared) para el segundo árbol de regresión ('dtree2').**

# In[125]:


#Metricas de evaluación (RMSE y R^2)
print(np.sqrt(mean_squared_error(y_train,tr2))) 
print(r2_score(y_train, tr2))


# In[126]:


#Metricas de evaluación (RMSE y R^2)
print(np.sqrt(mean_squared_error(y_test,y2))) 
print(r2_score(y_test, y2)) 


# Los resultados para el modelo 'dtree2' muestra que el RMSE es 4197.658 para los datos de entrenamiento y 5065.194 para los datos de prueba. El valor R-cuadrado es 86.83 por ciento para los datos de entrenamiento y 81.37 por ciento para los datos de prueba. Este modelo con respecto a los anteriores muestra una mejora significativa. Esto muestra que el modelo de árbol de regresión con el parámetro 'max_depth' de 5 es el que funciona mejor.

# ## Random Forest
# Para mejorar aun más el modelo de regresion se emplea un algoritmo de Random Forest.

# In[127]:


#Iniciando el modelo Random Forest con 500 arboles
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
#Ajuste del modelo a los datos de entrenamiento.
model_rf.fit(X_train, y_train)
#Predicción de los datos de entrenamiento
pred_train_rf= model_rf.predict(X_train)
#Metricas de evaluación
print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
print(r2_score(y_train, pred_train_rf))


# In[128]:


#Prediccion de los datos de prueba
pred_test_rf = model_rf.predict(X_test)
#Metricas de evaluación
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(y_test, pred_test_rf))


# Los resultados anteriores muestran que los valores RMSE y R^2 en los datos de entrenamiento son 1608.444 y 98.06 por ciento, respectivamente. Para los datos de prueba, el resultado de estas métricas son 4122.63 y 87.66 por ciento, respectivamente. El rendimiento del modelo de random forest resulto ser superior al de los modelos de árboles de decisión creados anteriormente.

# ## Empleo del modelo para predicción de nuevos datos 
# A continuación se emplea el modelo para predecir los precios de "test_price" usando el modelo de random forest.

# In[138]:


#Iniciando el modelo Random Forest con 500 arboles
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
#Ajuste del modelo a los datos de entrenamiento.
model_rf.fit(X, y)
#Prediccion de los nuevos datos
pred_toTest_rf = model_rf.predict(X_toTest)
#Metricas de evaluación
print(np.sqrt(mean_squared_error(y[500:1500],pred_toTest_rf)))
print(r2_score(y[500:1500], pred_toTest_rf))


# **La R^2 negativa esta indicando qu el modelo elegido no sigue la tendencia de los datos, se ajusta peor que una línea horizontal.**
# El modelo necesita revisarse con mayor detalle evaluando las variables categóricas.

# In[139]:


#Exportación de los resultados
df = pd.read_csv('Ejercicios_entrevista/test_price.csv')
df['precio predecido'] = np.rint(pred_toTest_rf)
df.to_csv('pred_price.csv')


# In[ ]:




