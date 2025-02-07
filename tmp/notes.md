## Reunión 6 - 2024-MM-DD (Fecha a ser añadida)


### Temas Discutidos:

1.  **Uso de `decision_function` en Isolation Forest:**

    *   Carlos recomienda utilizar `decision_function` en lugar de `predict` al trabajar con Isolation Forest.
    *   `decision_function` proporciona una puntuación continua entre 1 y -1, lo cual permite una mayor flexibilidad para definir un umbral de anomalía.
    *   En Isolation Forest, el flujo de trabajo sería:
        *   `Fit` (Entrenamiento del modelo)
        *   `Decision-function` (Obtención de las puntuaciones de anomalía)
    *   La principal ventaja es obtener un array continuo de puntuaciones en lugar de etiquetas binarias.

2.  **Concatenación de Canales AIA:**

    *   Es fundamental combinar todos los canales de AIA en una sola variable.
    *   El proceso implica concatenar los canales uno detrás de otro.
    *   Se debe realizar un "flatten" únicamente en la dimensión espacial, manteniendo la dimensión de canales intacta.
    *   La estructura de datos resultante debe ser de dimensiones `(512x512)x9`, donde cada píxel conserva sus 9 canales.

3.  **Enfoque de Normalizing Flows (NFlows):**

    *   Para NFlows, se debe tratar cada píxel de forma independiente.
    *   El objetivo es que la red neuronal aprenda qué se considera "normal" a nivel de píxel individual.
    *   El código de preparación de datos (`preparation data`) debe ser idéntico al utilizado para otros métodos (como Isolation Forest).
    *   En NFlows, en lugar de trabajar con imágenes completas, se deben usar los mismos datos preparados que se utilizarían en otros algoritmos.
    *   Se sugiere utilizar la rutina `prep` de Sunpy para el preprocesamiento de datos. Esta rutina toma los canales como entrada y devuelve los datos arreglados (centrados, con el tamaño de píxel adecuado, etc.).
    *   Carlos menciona la librería `einops` como potencialmente útil para la manipulación de datos.
    *   Se recomienda enfocarse principalmente en Isolation Forest y Normalizing Flows.

### Tareas

#### Algoritmos:

1.  **Isolation Forest:**
    *   Implementar el flujo de trabajo utilizando `decision_function` en lugar de `predict`.
    *   Experimentar con diferentes umbrales en las puntuaciones de `decision-function` para definir anomalías.

2.  **Normalizing Flows (NFlows):**
    *   Adaptar el código de NFlows existente para trabajar con los datos preparados (píxeles individuales con canales concatenados).
    *   Generar visualizaciones (plots) para evaluar el rendimiento del modelo NFlows.

#### Datos:

1.  **Preparación de Datos:**
    *   Asegurar la concatenación correcta de todos los canales de AIA.
    *   Implementar el "flatten" espacial para obtener la estructura de datos `(512x512)x9`.
    *   Verificar la correcta aplicación de la rutina `prep` de Sunpy para el preprocesamiento.
    *   Considerar la librería `einops` para facilitar la manipulación y reestructuración de los datos si es necesario.

#### Umbrales y Métricas:

1.  **Definición de Umbrales:**
    *   Experimentar y definir umbrales adecuados para las puntuaciones de `decision-function` en Isolation Forest y, si es aplicable, en Outlier Factor.
    *   Justificar la selección de umbrales basados en los resultados y el contexto del problema.

2.  **Evaluación y Comparación:**
    *   Comparar el rendimiento de Isolation Forest y Normalizing Flows en la detección de anomalías.
    *   Utilizar métricas relevantes para evaluar la efectividad de cada método.

### Próximos Pasos

1.  **Implementar `decision-function` en Isolation Forest:** Modificar el código de Isolation Forest para usar `decision-function` y experimentar con umbrales.
2.  **Preparar datos concatenados:**  Asegurar la correcta preparación de los datos con canales concatenados y estructura `(512x512)x9`.
3.  **Adaptar NFlows y generar plots:**  Ajustar el código de NFlows para trabajar con los datos preparados y generar visualizaciones relevantes para evaluar el modelo.
4.  **Experimentar con Outlier Factor:**  Explorar la posibilidad de utilizar Outlier Factor de manera similar a Isolation Forest, empleando un método que permita obtener un rango continuo de valores y ajustar umbrales.
5.  **Considerar Autoencoders:**  Si se dispone de tiempo y los resultados con Isolation Forest y NFlows son satisfactorios, explorar la implementación de Autoencoders para la detección de anomalías.
6.  **Documentar el proceso y resultados:**  Mantener una documentación clara del proceso seguido, los resultados obtenidos y las decisiones tomadas en cada paso.

---








Algunas anotaciones:
Carlos en el isolation forest no usaba el predict, sino que usaba el decisión-function que va entre 1 y -1 porque te da la libertad de definir un threshold de que es lo que se considera una anomalía y que se consideran pídeles normales.
Isolation forest
Fit
Decision-function
La única diferencia es que ese array es continuo

Hay que mezclar todos los canales de AIA. Una variable que concatene todos los canales uno detrás del otro. El flatten se hace solo en la dimensión espacial, es decir que cada pixel sigue teniendo los 9 canales. Entonces las dimensiones son (512x512)x9

En en Nflows, no miras imágenes chiquitas sino que tratas todos los pixeles independientemente  para que la red aprenda que es lo normal. El código que trata las imágenes, es decir el preparation data sea exactamente el mismo. Es decir que en flow en lugar de coger imágenes coja los mismos datos que usarían los otros métodos.
Sunpy tiene la rutina prep que en principio como input mete los canales y te escupe los datos arreglados, es decir, centrados y tamaño de pixels y etc.
Librería para usar que puede ser funcional https://github.com/arogozhnikov/einops
Carlos sugiere enfocarse en isolation forest y el normalizing flows.


Enfatizar y que quede claro: 1. queremos todos los canales concatenados. 2. Tanto con el isolation forest como con el outlier factor, es decir los dos algoritmos, saltar de la forma predict a la forma en la que puedas tener todo el rango de valores y tu jugar con ese umbral. 3. Carlos comparte el nflows que tienes que adaptar a los datos y tener un par de plots. 4. Si todo está hecho y hay tiempo probar con el autoencoder. 