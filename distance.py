import numpy as np
'''
Created on Oct 19, 2019
@author: María José Briceño
'''
features = np.fromfile('features.np', dtype=int)
test_features = np.fromfile('featuresTest.np', dtype=int)
etiquetas = np.fromfile('labels.np', dtype=int)

features_matrix = np.reshape(features,(7000,512))
test_features_matrix = np.reshape(test_features,(3000,512))

# distancia euclidiana
distancia_euclidiana = []

# calculamos la distancia euclidiana
for row_a in test_features_matrix:
    for row_b in features_matrix:
     dist = np.linalg.norm(row_a-row_b)
     distancia_euclidiana.append(dist)

# Agregar las etiquetas para ordenar por etiqueta y distancia :)

# ordenar el arreglo de menor a mayor
distancia_euclidiana = distancia_euclidiana.sort()




