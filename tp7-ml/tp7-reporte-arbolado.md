# Reporte arbolado

## Modelos utilizados

### R
Se intento con rpart, ranger, caret, xgboost, gbm, randomForest, pero no se pudo obtener un resultado satisfactorio.

### Python

Se intentó con pytorch las siguientes arquitecturas secuenciales:

- 1 capa oculta de 128 neuronas
- 2 capas ocultas de 128 neuronas con y sin dropout
- 10 capas ocultas de 16 neuronas con y sin dropout
- 2 capas ocultas de 1024 neuronas con y sin dropout

Se probaron distintos learning_rate:

- 0.01
- 0.0001
- 0.00001
- 0.000005

Se experimento con distintos batch_size:

- 8
- 16
- 32
- 64

Se probaron varios metodos para lidiar con las clases desbalanceadas

- class_weight: Usando BCEWithLogitsLoss (Con diferentes pesos para cada clase)
- oversampling
- undersampling
- bootstrapping
- bootstrapping + oversampling


Las metricas utilizadas fueron:

- accuracy
- bce
- auc

Se utilizó un 80% del dataset para entrenamiento y un 20% para validacion.

#### Columnas

Se utilizaron todas las columnas del dataset, excepto las que no aportaban informacion (id, ultima_modificacion)

Las columnas categoricas se convirtieron a one-hot-encoding.

Los encoders fueron entrenados en el dataset completo, ya que existian especies que no estaban presentes en el dataset de entrenamiento, pero sí en el de test.

También se experimento con feature engineering:

- Se agregó una columna que indicaba la "peligrosidad" esperada de una especie, basado en la cantidad de ejemplares de esa especie que tenian inclinacion peligrosa en el conjunto de entrenamiento.
- Se agregó una columna similar para otras variables categoricas - como la seccion.

#### Resultados

Los mejores modelos fueron submiteados, pero no se obtuvo un resultado satisfactorio.

En el mejor de los casos logrando 69 auc en el set de validacion.

![img_8.png](img_8.png)
