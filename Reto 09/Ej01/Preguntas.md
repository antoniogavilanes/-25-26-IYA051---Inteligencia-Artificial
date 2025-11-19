# Preguntas Ej03

## ¿Qué tipo de problema de aprendizaje automático se aborda en este ejercicio y por qué se clasifica así?

Es un problema de clasificación supervisada, porque el objetivo es predecir una categoría discreta —en este caso, si un paciente tiene o no tiene diabetes— a partir de un conjunto de atributos de entrada.
La variable objetivo es binaria: 0 (no diabetes) y 1 (diabetes).

## ¿Qué representan los 8 atributos de entrada y qué tipo de variable es la etiqueta “Categoría”?

Los 8 atributos son características clínicas del paciente, por ejemplo:
<ul>
<li>Embarazos</li>
<li>Glucosa</li>
<li>Presión sanguínea</li>
<li>Pliegue cutáneo</li>
<li>Insulina</li>
<li>Índice de masa corporal</li>
<li>Función pediátrica (o similar, según dataset)</li>
<li>Edad</li>
</ul>
Son variables numéricas utilizadas para estimar el riesgo de diabetes.
La etiqueta “Categoría” es una variable cualitativa binaria (0 o 1), usada como objetivo del modelo.

## ¿Por qué es importante dividir el conjunto de datos en entrenamiento y test? ¿Qué proporciones se usan aquí?

Dividir el dataset permite entrenar el modelo con una parte y evaluarlo con datos nuevos.
Si entrenaras y evaluaras con el mismo conjunto, el modelo podría parecer “perfecto” solo porque ya ha visto todos los datos.
Aquí se usa:

<li>60 % para entrenamiento</li>
<li>40 % para test</li>

Además, se aplica stratify=y para mantener la proporción de clases.

## ¿Cómo afecta el valor de k en el comportamiento del clasificador kNN? ¿Qué implicaciones tiene un valor muy bajo o muy alto?, ¿Qué significa que la máxima precisión se obtenga con k = 7?
## Interpreta la matriz de confusión obtenida. ¿Qué representan los valores VP, VN, FP y FN en el contexto de la diabetes?
## Calcula e interpreta las métricas de precisión, recall y f1-score para cada clase. ¿Qué conclusiones puedes extraer?
## ¿Qué indica el valor del área bajo la curva ROC (AUC = 0.7345) sobre el rendimiento del modelo?
## Si se modifica el porcentaje de datos de test a 20 %, ¿cómo crees que afectaría al modelo?
## ¿Qué predice el modelo para el nuevo paciente con los atributos dados? ¿es confiable esta predicción? Explica por qué y en qué te basas.

