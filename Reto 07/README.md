# Reto 07

Tema: ML: Regresión polinómica

En "Materiales de la asignatura" > "Machine Learning" > "Regresión"

Ahí hay 5 carpetas con ejercicios a trabajar con sus códigos. Además de hacer que funcionen en un cuaderno de Google Colab, con los correspondientes comentarios de haber entendido el código, hay que responder las siguientes cuestiones:

## Para el Ej01:

# 1.1 ¿Cuál es el objetivo principal de la regresión lineal en este ejercicio?
El objetivo es encontrar la relación lineal que mejor explique cómo varía el CO₂ en función de la anomalía de temperatura, ajustando una recta que permita realizar predicciones sobre el CO₂ a partir de nuevos valores de temperatura.

# 1.2 ¿Qué representan las variables X e y en el código y cómo se formalizan (tipo de estructura de datos)?

X representa la variable independiente: la anomalía de temperatura.
y representa la variable dependiente: la concentración anual de CO₂.

Ambas se formalizan como estructuras tipo DataFrame de Pandas y posteriormente como arrays bidimensionales de NumPy con forma (n,1), ya que la librería de regresión de sklearn requiere este formato.

# 1.3 ¿Qué hace el siguiente código?

regr = LinearRegression()

regr.fit(X_train, y_train)

Crea un modelo de regresión lineal vacío.

Entrena el modelo utilizando los datos de entrenamiento para calcular la pendiente y el intercepto que mejor ajustan la relación entre temperatura y CO₂.


# 1.4 ¿Qué significan los parámetros coef_ e intercept_ del modelo?

coef_ es la pendiente de la recta: indica cuánto aumenta el CO₂ por cada unidad que aumenta la temperatura.

intercept_ es el valor del CO₂ cuando la temperatura es cero; es el punto donde la recta corta el eje vertical.

# 1.5 Ecuación matemática del modelo de regresión obtenida. Si vuelvo a ejecutar el código, ¿varían los coeficientes de la ecuación? ¿por qué?

La ecuación resultante tiene la forma:

y=b0+b1X

donde:

b0= intercept_
b1= coef_

Si el código se ejecuta nuevamente con los mismos datos, los coeficientes no cambian, porque la regresión lineal es un método determinista. Solo variarían si cambian los datos o la división entre entrenamiento y test.

# 1.6 ¿Qué hace el siguiente bloque de código y qué representa la gráfica resultante? ¿qué diferencia hay entre y_train e y_test, y por qué se separan estos dos tipos de datos?

plt.scatter(X_train, y_train, color="red")

plt.scatter(X_test, y_test, color="blue")

plt.plot(X_train, regr.predict(X_train), color="black")

Muestra los datos de entrenamiento (rojo) y de test (azul).

Dibuja la recta del modelo entrenado (negro).

Diferencia entre y_train e y_test:

y_train se usa para entrenar el modelo.

y_test evalúa cómo generaliza el modelo a datos que no ha visto.

Se separan para evitar que el modelo memorice los datos y para medir su capacidad real de predicción.


# 1.7 ¿Qué miden las métricas MSE y R^2 que aparecen en el código?

MSE (Mean Squared Error): mide el error promedio cuadrático entre valores reales y predicciones. Cuanto menor, mejor.

R² (Coeficiente de determinación): indica qué porcentaje de la variabilidad del CO₂ está siendo explicado por la temperatura. Valores cercanos a 1 indican un ajuste muy bueno.

# 1.8 Explica los resultados de R^2 en entrenamiento y de test.

Un R² alto en entrenamiento indica que el modelo ajusta bien los datos con los que aprendió.

Un R² alto en test indica que el modelo también predice correctamente valores nuevos.
Si ambos valores son parecidos, el modelo generaliza bien; si difieren mucho, puede haber sobreajuste o mala generalización.

# 1.9 ¿Cómo se haría una predicción nueva con el modelo entrenado? Formalízalo en código con un ejemplo.

nuevo_valor = [[0.8]]  # anomalía de temperatura
prediccion = regr.predict(nuevo_valor)
print(prediccion)

# 1.10 ¿Qué parámetros o configuraciones se podrían cambiar para mejorar el modelo?

Normalizar o estandarizar las variables.
Añadir nuevas características (más variables relacionadas).
Ampliar el conjunto de entrenamiento.
Probar modelos no lineales (regresión polinómica).
Ajustar el rango temporal o eliminar valores atípicos.

## Para el ejercicio 02:

# 2.1 ¿Qué propósito tiene la función generador_datos_simple()?

Crear un conjunto de datos sintético que siga una relación lineal básica del tipo

y=βX+ruido
y=βX+ruido

permitiendo generar valores aleatorios de X y sus correspondientes etiquetas Y.
Sirve para simular datos reales con los que entrenar y evaluar un modelo de regresión lineal.

# 2.2 ¿Por qué se introduce un término de error aleatorio en la generación de datos?

Para simular el ruido presente en datos reales.
Sin ruido, la relación sería perfectamente lineal y el modelo obtendría siempre un ajuste perfecto, lo cual no representa situaciones del mundo real.

# 2.3 ¿Qué papel tien el parámetro beta en la simulación?

beta es la pendiente real de la relación entre X y Y.
El modelo de regresión intentará recuperar ese valor durante el entrenamiento.
Si beta = 10, la relación ideal es 
y≈10X
y≈10X, más el ruido.


# 2.4 ¿Por qué se realiza la división de los datos 70% / 30% en entrenamiento y test? ¿harías otra división? ¿en función de qué se cogen esos porcentajes?

Se divide el dataset en:

70% para entrenar, suficiente para que el modelo aprenda.

30% para test, suficiente para evaluar cómo generaliza.

Se podría usar otro reparto (80/20, 75/25, 60/40…) dependiendo del tamaño del dataset.
Cuantos menos datos tengas, mayor porcentaje conviene reservar para entrenamiento.

# 2.5 ¿Qué información proporcionan los atributos coef_ e intercept_ después del entrenamiento? Semejanzas y diferencias respecto del código del ej01.

coef_: la pendiente estimada del modelo (cuánto cambia Y por cada unidad de X).

intercept_: la intersección con el eje Y (valor de Y cuando X=0).

Es lo mismo que en el Ej01.
La diferencia es que aquí los datos fueron generados artificialmente con un beta conocido, por lo que el modelo debería aproximarse a ese valor.

# 2.6 Cuánto vale R^2. Interprétalo y compáralo con el ej01.

El valor depende de la ejecución, pero en general:

R² suele ser moderado o bajo por la desviación = 200 introducida.
En el Ej01 (datos reales de CO₂ y temperatura), el ajuste suele ser más estable, ya que la relación es clara y el ruido no es tan extremo.
Aquí, el gran ruido hace que el modelo no pueda explicar tanta variabilidad.

# 2.7 ¿Por qué son diferentes los valores de R^2 del test y del entrenamiento? ¿Qué valores desearíamos tener en ellos?

Porque el modelo se ajusta únicamente con los datos de entrenamiento.
Si los datos de test tienen una distribución algo diferente (por aleatoriedad), el rendimiento suele variar.
Lo ideal sería que, ambos R² sean altos y muy parecidos entre sí, lo que indicaría buena capacidad de generalización.

# 2.8 ¿Qué pasaría si aumentamos el parámetro desviacion en el generador de datos? ¿para qué querríamos hacer esto?

El ruido aumenta, los puntos están más dispersos.
La relación lineal se vuelve más difícil de aprender.
El error crece y R² disminuye.

Aumentarlo sirve para simular datos más “reales”, analizar la robustez del modelo ante ruido.

# 2.9 ¿Por qué el código hace reshape((muestras,1)) al generar X e y?

Porque sklearn espera que la entrada sea una matriz de dos dimensiones, aunque solo haya una variable.
Sin reshape, serían arrays unidimensionales y el modelo no los aceptaría.

# 2.10 Si yo hago X=50, ¿qué significaría respecto al ejemplo y al modelo calculado?

Significa introducir un nuevo valor para predecir su etiqueta según la ecuación aprendida.
El modelo devolverá algo cercano a:

y≈50⋅β(mas el error de ajuste)

En este caso, se está prediciendo el valor de Y cuando X=50.

## Para el Ejercicio 03:

# 3.1 Diferencia entre regresión lineal simple de ejercicios anteriores y la múltiple de este.

La regresión lineal simple utiliza una única variable independiente para predecir una salida, mientras que la regresión lineal múltiple utiliza varias variables independientes simultáneamente.
En este ejercicio, se pasa de un único predictor (X) a tres (temperatura, nivel del mar y masa glaciar), lo que permite modelar relaciones más complejas.

# 3.2 Ecuación del modelo obtenido. ¿Qué significa el término independiente de la ecuación? (a nivel físico del caso de uso y a nivel matemático)

A nivel físico: representa la concentración estimada de CO₂ cuando todas las variables (temperatura, nivel del mar y masa glaciar) fuesen cero. No es un valor físicamente realista, pero actúa como referencia del modelo.

A nivel matemático: es el punto donde la recta (o hiperplano) corta el eje de la variable dependiente. Ajusta verticalmente el modelo para minimizar el error.

# 3.3 ¿De cuántas variables de entrada depende la salida? ¿Podríamos hacerlo de una sola? ¿de qué depende?

La salida depende de tres variables de entrada: temperatura, nivel del mar y masa glaciar.

¿Podríamos hacerlo con una sola?
Sí,bastaría elegir una (por ejemplo, temperatura).
¿De qué depende?
Depende de:

La correlación entre esa variable y el CO₂.
El objetivo del análisis.
La complejidad que queramos capturar.

Con una sola variable, el modelo sería más simple, pero también menos informativo.

# 3.4 ¿Qué significa que el coeficiente de la masa glaciar sea negativo en este ejercicio?

Un coeficiente negativo indica que, según el modelo, a menor masa glaciar, mayor concentración de CO₂, y viceversa.
Esto tiene sentido físico: la pérdida de masa glaciar es un efecto asociado al calentamiento global, que a su vez está relacionado con niveles crecientes de CO₂.

El modelo refleja esa relación inversa.

# 3.5 Interpreta los valores obtenidos de R^2 en entrenamiento y test

R² entrenamiento: mide qué porcentaje de la variabilidad del CO₂ explica el modelo usando los datos con los que aprendió. Un valor alto indica buen ajuste.

R² test: evalúa si el modelo generaliza correctamente a datos nuevos.
Si R² test es similar al de entrenamiento, el modelo no está sobreajustado.
Si es mucho menor, significa que el modelo aprende patrones muy particulares del entrenamiento pero no generaliza bien.

# 3.6 Al aumentar el número de variables de entrada, ¿qué ventajas e inconvenientes tendría? Por ejemplo, si incluyésemos la deforestación.

Ventajas:

El modelo puede capturar relaciones más complejas.
Puede aumentar la capacidad predictiva si las variables añadidas están correlacionadas con la salida.
Se obtiene una visión más completa del fenómeno.

Inconvenientes:

Mayor riesgo de sobreajuste.
Requiere más datos para entrenarse correctamente.
Se pueden introducir variables irrelevantes o ruidosas.
Interpretar los coeficientes se vuelve más difícil.
Los modelos pueden volverse inestables si variables son muy correlacionadas entre sí (colinealidad).

Ejemplo: incluir deforestación podría ayudar si está bien medida, pero también podría añadir ruido si la serie no está bien sincronizada.

# 3.7 ¿Crees que es adecuada la regresión lineal múltiple para predecir CO2 en este caso? Explica por qué.

Es una aproximación válida, pero limitada.

Adecuada porque:

Permite una primera aproximación sencilla.
Captura tendencias generales.
Facilita interpretar la influencia de cada variable.

Limitada porque:

El cambio climático es un fenómeno altamente no lineal.
Factores como aerosoles, volcanes, circulación oceánica y uso de suelo no están incluidos.
Las relaciones entre variables no son puramente lineales.
El modelo puede simplificar demasiado el comportamiento real del CO₂.

Conclusión:
La regresión lineal múltiple sirve como herramienta introductoria, pero no como modelo predictivo robusto del CO₂ a nivel científico.

## Para el Ejercicio 04:

# 4.1 Diferencias entre regresión lineal y polinómica

La regresión lineal ajusta una recta (modelo de primer grado).
La regresión polinómica ajusta un polinomio de grado mayor (curvas), lo que permite modelar relaciones no lineales entre X e y.
En este ejercicio, la función real 
f(x)=xsin⁡(x)
f(x)=xsin(x) es claramente curva, así que la regresión lineal no serviría, mientras que los polinomios sí pueden aproximarla.

# 4.2 ¿Para qué se usa la función np.poly() en el código?

Grado 3: modelo relativamente flexible; captura parte de la forma, pero no todos los giros de la curva real.
Grado 4: más capacidad para seguir la función; mejora la aproximación.
Grado 5: aún más flexible y capaz de ajustarse mejor a la oscilación de 
xsin⁡(x)
xsin(x).

En general:
A mayor grado, más capacidad de ajuste, pero también mayor riesgo de sobreajuste si el grado fuera exagerado.

# 4.3 Explica la métrica utilizada para evaluar la calidad de ajuste de cada modelo polinómico

El ECM (error cuadrático medio) mide cuánto se desvía la predicción del valor real:

Si el ECM disminuye al aumentar el grado, significa que el polinomio se está ajustando mejor.

Normalmente:

Grado 3 tendrá el ECM más alto.
Grado 4 será intermedio.
Grado 5 tendrá el ECM menor porque es el que más se adapta a la curva real.

Conclusión: El polinomio de mayor grado está capturando mejor la forma ondulada de la función.

# 4.4 Por qué el modelo de grado 5 tiene menos error que los de grado 3 y 4

Ventajas:

Ajustaría prácticamente todos los puntos de entrenamiento.

Problemas:

Sobreajuste extremo: la curva oscilaría exageradamente.
Mala generalización: valores fuera de los puntos de entrenamiento serían muy erráticos.
Inestabilidad numérica en polyfit.

En otras palabras, acabarías con una curva que parece dibujada por alguien que tomó demasiado café.

# 4.5 Explica el overfitting o sobreajuste en el contexto del ejemplo.

Porque la función real 
xsin⁡(x)
xsin(x) es suave, continua y relativamente regular en el intervalo [0, 10].
Los polinomios de grado 3–5 tienen suficiente flexibilidad para aproximarla sin llegar a comportarse de forma inestable.

# 4.6 ¿Para qué se usa la función np.polyval() en el código? Diferencias con np.poly() ¿Por qué empiezan por np. ambas?

El modelo de grado 5 genera una estimación aproximada de:

y(6)≈6sin⁡(6)
y(6)≈6sin(6)

Lo cual coincide bastante con la función real.
Esto indica que el modelo de grado 5 generaliza correctamente dentro del rango donde se entrenó (0–10).

Si lo usaras fuera de ese rango… bueno, ahí ya empiezan los temblores del sobreajuste.

# 4.7 Si siguiésemos aumentando el grado del polinomio del modelo que efectos podrían observarse. ¿Con qué grado te quedarías tú y por qué?

El polinomio de grado 5, porque:

Tiene el menor ECM.
Sigue de forma más natural la forma ondulada de la función.
No muestra oscilaciones artificiales.

Es un término medio razonable entre flexibilidad y estabilidad.

## Para el Ejercicio 05: Resolverlo según lo inferido y aprendido en los ejemplos anteriores.
