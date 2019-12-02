import numpy as np

'''
Created on Oct 19, 2019
@author:    María José Briceño
            Guillermo Otarola
'''
label_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/labels.np'
label_test_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/labelsTest.np'
features_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/features.np'
features_test_path = 'C:/Users/mjnik/Desktop/Programas Python/computer_vision/Sketch_EITZ/featuresTest.np'

features = np.fromfile(features_path, np.float32)
test_features = np.fromfile(features_test_path, np.float32)
labels = np.fromfile(label_path, dtype =int)
test_labels = np.fromfile(label_test_path, dtype=int)

features = np.reshape(features, (14000, 512))
test_features = np.reshape(test_features, (6000, 512))

labels = np.reshape(labels, (14000, 1))
test_labels = np.reshape(test_labels, (6000, 1))


# distancia euclidiana
distancia_euclidiana = np.array([])
etiqueta_base = np.array([])

# calculamos la distancia euclidiana
# for row_a in test_features:
row_a = test_features[222]
label_a = test_labels[222]




# Calcula el vector de distancia entre la imagen de consulta versus las imagenes de la base de datos
for row_b in features:
    dist = np.linalg.norm(row_a - row_b)
    distancia_euclidiana = np.append(distancia_euclidiana, dist, axis=None)

# Arreglo de etiquetas
# Buscamos el numero total de ground truth
ground_truth = 1
for row_c in labels:
    if row_c == int(label_a):
        ground_truth = ground_truth +1
    etiqueta_base = np.append(etiqueta_base, row_c)

# copio el arreglo
aux_euclidiano = np.copy(distancia_euclidiana)

# Ordeno de menor a mayor de acuerdo a las distancias
aux_euclidiano.sort()

# Creo un arreglo de resultados
result = np.array([])

# Escojo con cuantas imagenes quiero comparar la imagen de consulta con las de la BD
numero_imagenes = 11

# Busco las posiciones de las etiquetas de acuerdo a los valores de las distancias
for i in range(0,numero_imagenes):
    donde = np.where(distancia_euclidiana == aux_euclidiano[i])
    result = np.append(result, donde)

# Almaceno el valor de las etiquetas en este arreglo
average_precision = 0
contador = 0
positivo = 0

# Busco el contenido de en el arreglo de las etiquetas de acuerdo a los resultados obtenidos anteriormente
# Extraigo el valor de la etiqueta
# Comparo con el valor de la etiqueta de la query con el valor de la etiqueta que encontre
# Si es igual lo cuento y lo sumo de acuerdo al criterio de Average Precision
for row in result:
    contador = contador + 1
    fila = int(row)
    valor_etiqueta = int(etiqueta_base[fila])

    # Si la etiqueta de la consulta es igual a la de la clase
    if int(label_a) == valor_etiqueta:
        positivo = positivo + 1
        average_precision =(positivo / contador) + average_precision

# Imprimo average precision
print("average_precision ", average_precision/numero_imagenes)

# Calculo de mAP
# AP / Nº total de queries
mean_average_precision = average_precision/test_features.size
print("mAP ", mean_average_precision)

