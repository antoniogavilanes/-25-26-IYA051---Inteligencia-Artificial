# Preguntas Ej03

## ¿Qué tipo de problema de aprendizaje automático se está abordando en este caso y cuál es la variable objetivo?

Es un problema de clasificación supervisada.
Más concretamente, clasificación binaria, porque solo hay dos posibilidades:
<ul>
<li>1: la canción llegó al Nº1</li>
<li>0: no llegó</li>
</ul>

La variable objetivo (y) es la columna top, que indica si el artista o canción alcanzó el puesto Nº1.

## ¿Por qué es necesario dividir los datos en variables de entrada X y variable objetivo y antes de entrenar el modelo?

Porque el modelo necesita:

<ul>
<li>X: la información con la que aprende (duración, género, streams, etc.).</li>
<li>y: lo que queremos que el modelo prediga.</li>
</ul>

Si no hacemos esa separación, el modelo estaría aprendiendo con su respuesta “pegada”…
Vamos, sería como copiar en un examen mirando el solucionario: aprende todo menos lo importante.

## ¿Qué función cumplen los parámetros criterion, min_samples_split, min_samples_leaf, max_depth y class_weight al crear el árbol de decisión?

Estos parámetros controlan cómo crece y se ajusta el árbol.

<strong>criterion</strong>
Define la métrica para medir la calidad de una división.
Puede ser "gini" o "entropy".
Le dice al árbol cómo decidir “¿esta división es buena o mala?”.

<strong>min_samples_split</strong>
Número mínimo de muestras necesarias para dividir un nodo.
Evita que el árbol divida por dividir como si estuviera aburrido.

<strong>min_samples_leaf</strong>
Mínimo de muestras que debe tener una hoja del árbol.
Esto evita hojas con 1 muestra, que serían puro sobreajuste.

<strong>max_depth</strong>
Profundidad máxima del árbol.
Limita que el árbol se vuelva “demasiado listo” para los datos de entrenamiento y luego fracase en los de prueba.

<strong>class_weight</strong>
Pondera las clases para combatir el desbalanceo.
Si una clase aparece muy poco, se le da más importancia para que el modelo no la ignore.

## ¿Por qué se utiliza el parámetro class_weight={1:3.5} en este ejercicio?

Porque seguramente las canciones que llegan al Nº1 son minoría en el dataset.
Si no se ajusta el peso entonces el modelo puede aprender a predecir siempre “No llega al Nº1” u obtiene una precisión decente pero no sirve para nada.

Al poner {1: 3.5}, le decimos “Ey, cuando te equivoques prediciendo un Nº1, cuenta como 3.5 errores.”
Esto empuja al modelo a prestar atención a esa clase.

## ¿Qué significa la precisión (accuracy) del modelo y cómo se calcula en el código?

La precisión es la proporción de predicciones correctas sobre el total.

## ¿Qué diferencia hay entre los métodos predict() y predict_proba()?

<strong>predict()</strong>
Devuelve directamente la clase final: 0 o 1, sin explicaciones.
Es el típico amigo que no da contexto: “Sí”, “No”, y ya.

<strong>predict_proba()</strong>
Devuelve las probabilidades de cada clase.

## ¿Qué implicaciones tiene que el árbol tenga una profundidad máxima (max_depth = 4)??

Un árbol menos profundo significa:
<ul>
<li>Generaliza mejor (menos sobreajuste).</li>
<li>Aprende patrones más simples y robustos.</li>
<li>Es más interpretables, no una jungla de ramas.</li>
</ul>
Pero también:
<ul>
<li>Podría perder precisión si el problema es muy complejo.
(Como un camarero que solo sabe hacer gin-tonic y mojitos: fiable, pero limitado).</li>
</ul>
En definitiva, max_depth=4 evita que el árbol se vuelva un monstruo gigante que memoriza el dataset.
