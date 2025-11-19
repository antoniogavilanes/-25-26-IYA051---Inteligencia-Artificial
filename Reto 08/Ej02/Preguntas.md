# Preguntas Ej02

## ¿Qué tipo de problema de aprendizaje automático se está resolviendo y por qué?

Es un problema de clasificación supervisada.

Supervisada: porque el dataset tiene etiquetas conocidas (el tipo de flor).
Clasificación: porque el modelo debe asignar una categoría entre 3 posibles especies: setosa, versicolor y virginica.

Aquí no predecimos valores numéricos, sino clases, así que es clasificación pura y dura.

## ¿Qué representa el dataset “Iris” y qué tipo de variables contiene?

El dataset Iris es uno de los conjuntos de datos clásicos en Machine Learning.
Contiene mediciones de características físicas de flores del género Iris.

CONTENIDO DEL DATASET

150 muestras
3 especies:
Iris setosa
Iris versicolor
Iris virginica

VARIABLES

Todas son numéricas continuas:
Longitud del sépalo (cm)
Anchura del sépalo (cm)
Longitud del pétalo (cm)
Anchura del pétalo (cm)

## ¿Qué papel cumple el método fit() en el modelo de árbol de decisión?

fit() es literalmente el entrenamiento del modelo.

Toma los datos (X)
Toma las etiquetas (y)
Construye el árbol creando divisiones (splits) que reduzcan la impureza
Genera una estructura final que usará para clasificar datos nuevos

En otras palabras:
fit() es donde el árbol "aprende" de los ejemplos.

## ¿Por qué es importante visualizar el árbol de decisión generado con graphviz?

Porque te permite:
<ul>
<li><strong>Interpretar el modelo</strong>
Puedes ver cómo el árbol decide según las variables.</li>
<li><strong>Detectar sobreajuste</strong>
Si el árbol parece la raíz de Yggdrasil (infinitas ramas), está overfitted.</li>
<li><strong>Explicar decisiones</strong>
Muy útil si presentas resultados y necesitas justificar “por qué el modelo dijo eso”.</li>
</ul>
Los árboles son de los modelos más interpretables, y Graphviz te lo pone bonito y ordenado para que no te explote la cabeza.

## ¿Qué significa el resultado que devuelve la función predict() en este contexto?
predict() recibe una medición de flor (las 4 características) y devuelve:

un número (0, 1 o 2)

que corresponde a una especie en iris.target_names

Ejemplo:
```
pred1 = clf.predict([[7,3,5,1]])
```
Esto podría devolver 2, que significa:
```
iris.target_names[2] -> 'virginica'
```
Es decir, el modelo cree que esas medidas corresponden a una Iris virginica.

## ¿Qué ventajas y desventajas tienen los árboles de decisión frente a kNN algoritmos de clasificación?

<strong>Ventajas de Árboles de Decisión</strong>
<ul>
<li>Muy fáciles de interpretar</li>
<li>Inferencia rápida, clasificar es instantáneo</li>
<li>Pueden manejar datos categóricos y numéricos</li>
<li>Necesitan muy poca normalización o preprocesado</li>
<li>Detectan interacciones entre variables automáticamente</li>
</ul>

<strong>Desventajas</strong>
<ul>
<li>Pueden sobreajustarse si no se podan</li>
<li>Sensibles a pequeños cambios en los datos</li>
</ul>

<strong>Ventajas de kNN</strong>
<ul>
<li>Simple de entender y programar</li>
<li>Funciona muy bien cuando las clases están agrupadas espacialmente</li>
<li>No necesita entrenamiento</li>
</ul>

<strong>Desventajas</strong>
<ul>
<li>Clasificar es lento con muchos datos</li>
<li>Requiere normalizar variables</li>
<li>Sufre en dimensiones altas</li>
<li>No da interpretabilidad</li>
</ul>

## Si cambiáramos los valores de entrada (por ejemplo, 4, 2, 1, 0.2), qué crees que ocurriría y por qué?

Esas medidas:
<ul>
<li>Sépalo pequeño</li>
<li>Pétalo MUY corto y estrecho: 1 cm y 0.2 cm</li>
</ul>
Eso es prácticamente la firma de Iris setosa, que tiene pétalos más pequeños que las otras especies.

Entonces el modelo devolvería:
<ul>
<li>setosa con alta probabilidad</li>
</ul>

<strong>¿Por qué?</strong>
Porque en el dataset, setosa ocupa claramente la región de valores bajos de pétalo.
Un árbol de decisión lo detecta enseguida y hace un split muy temprano en esa variable
