# Preguntas Ej02

## ¿Qué objetivo persigue el botánico con la aplicación del modelo kNN en este caso?

El botánico busca clasificar automáticamente una flor desconocida dentro de una de las tres especies del conjunto Iris.
En otras palabras: quiere que el modelo aprenda a reconocer patrones morfológicos (medidas de sépalos y pétalos) para que, cuando le entreguen una flor nueva, el sistema pueda decirle de qué especie es sin necesidad de análisis manual.

## ¿Qué variables componen el conjunto de datos Iris y qué tipo de problema de clasificación se plantea?

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

<ul>
<li>60 % para entrenamiento</li>
<li>40 % para test</li>
</ul>

Además, se aplica stratify=y para mantener la proporción de clases.

## ¿Cómo afecta el valor de k en el comportamiento del clasificador kNN? ¿Qué implicaciones tiene un valor muy bajo o muy alto?, ¿Qué significa que la máxima precisión se obtenga con k = 7?

<strong>k bajo (1, 2, 3):</strong>
El modelo se vuelve muy sensible al ruido. Tiende a sobreajustar.

<strong>k muy alto:</strong>
Empieza a ser demasiado general. Tiende a subajustar.

<strong>k = 7 como mejor resultado:</strong>
Significa que, para este dataset, un vecindario moderadamente amplio ofrece el mejor equilibrio entre variabilidad y estabilidad.
En otras palabras: ni tan pegajoso, ni tan distante.

## Interpreta la matriz de confusión obtenida. ¿Qué representan los valores VP, VN, FP y FN en el contexto de la diabetes?

Los valores VP, VN, FP y FN representan:
<ul>
<li><strong>VP (Verdaderos Positivos):</strong> pacientes con diabetes correctamente identificados.</li>
<li><strong>VN (Verdaderos Negativos):</strong> pacientes sin diabetes correctamente clasificados.</li>
<li><strong>FP (Falsos Positivos):</strong> el modelo predice diabetes cuando NO la hay.
En medicina, esto implica alarmar al paciente sin necesidad.</li>
<li><strong>FN (Falsos Negativos):</strong> el modelo dice “todo bien” cuando SÍ hay diabetes.
Este es el error más grave, porque retrasa un diagnóstico real.</li>
</ul>
La matriz muestra cómo se comporta el modelo en cada tipo de caso.

## Calcula e interpreta las métricas de precisión, recall y f1-score para cada clase. ¿Qué conclusiones puedes extraer?
<ul>
<li>Precisión (precision): proporción de predicciones positivas correctas.</li>
<li>Recall (sensibilidad): proporción de verdaderos positivos detectados.</li>
<li>F1-score: equilibrio entre precisión y recall.</li>
</ul>

Interpretación general:
<ul>
<li>La clase 0 suele tener mejores valores (porque es la más frecuente).</li>
<li>La clase 1 normalmente presenta menor recall → el modelo falla en detectar todos los casos reales de diabetes.</li>
<li>El F1-score de la clase 1 suele ser inferior → indica dificultad del modelo para distinguir completamente a los pacientes diabéticos.</li>
</ul>

<strong>Conclusión:</strong>
El modelo funciona aceptablemente, pero no es especialmente fuerte identificando a los pacientes con diabetes (clase minoritaria).

## ¿Qué indica el valor del área bajo la curva ROC (AUC = 0.7345) sobre el rendimiento del modelo?

Un AUC de 0.73 indica un desempeño moderado:
<ul>
<li>0.5 = clasificador aleatorio (tirar una moneda)</li>
<li>0.7–0.8 = aceptable</li>
<li>0.8–0.9 = bueno</li>
<li>0.9 = excelente</li>
</ul>

El modelo diferencia pacientes diabéticos y no diabéticos un 73% de las veces, lo que sugiere un rendimiento suficiente pero no sobresaliente.

## Si se modifica el porcentaje de datos de test a 20 %, ¿cómo crees que afectaría al modelo?

Usar un 20 % para test implicarían más datos para entrenar y el modelo podría aprender un poco mejor, también menos datos para test y causa que la evaluación es menos estable y sea más sensible al azar.

## ¿Qué predice el modelo para el nuevo paciente con los atributos dados? ¿es confiable esta predicción? Explica por qué y en qué te basas.

Relativamente, no demasiado:
<ul>
<li>El AUC es moderado (0.73).</li>
<li>La clase 1 suele tener bajo recall → el modelo no es muy bueno detectando casos de diabetes.</li>
<li>Una sola observación aislada nunca debe tomarse como diagnóstico real.</li>
</ul>