import numpy as np


'''
Created on Oct 19, 2019
@author: María José Briceño
'''
label_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/labels.np'
label_test_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/labelsTest.np'
features_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/features.np'
features_test_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/featuresTest.np'


features = np.fromfile(features_path, np.float32)
test_features = np.fromfile(features_test_path, np.float32)
labels = np.fromfile(label_path, dtype=int)
test_labels = np.fromfile(label_test_path, dtype=int)

features = np.reshape(features, (14000, 512))
test_features = np.reshape(test_features, (6000, 512))

labels = np.reshape(labels,(14000,1))
test_labels = np.reshape(test_labels,(6000,1))

etiqueta = np.array([])
for a in labels:
    etiqueta = np.append(etiqueta, a)

# distancia euclidiana
distancia_euclidiana = np.array([])

# calculamos la distancia euclidiana
# for row_a in test_features:
row_a = test_features[0]

for row_b in features:
    dist = np.linalg.norm(row_a-row_b)
    distancia_euclidiana = np.append(distancia_euclidiana, dist, axis= None)

# copio el arreglo
aux_euclidiano = np.copy(distancia_euclidiana)

# Ordeno de menor a mayor
aux_euclidiano.sort()

# Busco las posiciones de las etiquetas de acuerdo a los valores de las distancias
result0 = np.where(distancia_euclidiana == aux_euclidiano[0])
result1 = np.where(distancia_euclidiana == aux_euclidiano[1])
result2 = np.where(distancia_euclidiana == aux_euclidiano[2])
result3 = np.where(distancia_euclidiana == aux_euclidiano[3])
result4 = np.where(distancia_euclidiana == aux_euclidiano[4])
result5 = np.where(distancia_euclidiana == aux_euclidiano[5])


print(result0[0])
print(result1[0])
print(result2[0])
print(result3[0])
print(result4[0])
print(result5[0])

print(distancia_euclidiana)
print(aux_euclidiano)
# Calculo AP para los primeros 25 valores


# Busco los


#print(distancia_euclidiana[5])
#print(etiqueta[5])
#states=distancia_euclidiana==min(distancia_euclidiana )
#donde=np.where(states)[0]
#print(donde)
#print(etiqueta[2536])





