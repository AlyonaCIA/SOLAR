# Meeting Notes

## Meeting 1 - 2024-06-05

**Participants:** Alyona Carolina Ivanova Araujo, Professor Luis Felipe Giraldo Trujillo, Carlos José Díaz Baso, Juan Camilo Guevara mez

### Topics Discussed:

1. **Eventos atípicos:**

   - Eventos poco frecuentes que suceden en alguna localidad y tiempo en una combinación de canales.
   - Explicabilidad del modelo, caja negra, o interpretable.
   - Caja negra con interpretabilidad a posteriori, estudio de gradiente, preguntas como: ¿Cuánto se tiene que cambiar un input para que el resultado cambie?
   - Utilizar la información de interpretabilidad para motivar estudios en regiones específicas.

2. **Tipos de eventos atípicos:**

   - Diferentes clases de atípicos con diversos niveles de interés.

3. **Revisión de literatura (por Profesor Luis Felipe):**

   - "Anomaly detection" or "One class-classification".
   - Extreme Events / Extreme Value Theory.
   - Outlier detection.
   - Detección de anomalías para este problema y en general.
   - Self-supervised learning.

4. **Clarificación del problema y expectativas:**

   - Para nosotros, los eventos atípicos son el resultado. En otros casos, los eventos atípicos son los que se quieren quitar.
   - Incluye todo lo discutido, incluyendo el tiempo real.

5. **Estrategia de selección de datos:**

   - Empezar con pocos datos al principio.

6. **Recursos computacionales:**

   - Determinar el poder de cómputo disponible: computadora personal, otros.

7. **Código abierto:**

   - Decidir si el proyecto será open source y si se dejará el código y los pesos del modelo abiertos.

---

## Meeting 2 - 2024-07-10

**Participants:** Alyona Carolina Ivanova Araujo, Professor Luis Felipe Giraldo Trujillo, Carlos José Díaz Baso, Juan Camilo Guevara G.

### Topics Discussed:

1. **Tablita:**

   - **Escenario:**

     - Definición de contextos en los que se buscarán anomalías.

   - **Estructura de los datos:**

     - Descripción detallada de los tipos de datos y formatos que se manejarán.

   - **Criterios para saber cuáles son anomalías o no:**
     - Definición de umbrales y características que determinan qué se considera una anomalía.

2. **Plots:**

   - **Proof of Concept con imagen alrededor de un flare:**

     - Seleccionar una imagen de un flare solar como ejemplo.
     - Crear gráficos en 2D y 3D para demostrar la detección de anomalías.
     - Explicación de cómo los canales son casi ortogonales y su impacto en la detección.

   - **Clustering no supervisado:**
     - Aplicar clustering no supervisado a los datos.
     - Mostrar sobre imágenes del sol dónde aparecen los grupos encontrados.

3. **Requerimientos Adicionales y Comentarios del Profesor:**

   - Recursos de scikit-learn para detección de anomalías y clustering:
   - Detección de Outliers: Algunos ejemplos básicos pero interesantes con explicaciones detalladas.
   - Densidad: Posibilidad de generar un "mapa de calor" de manera sencilla y ver si hay algo relevante.
   - t-SNE para visualización: Usar t-SNE como estrategia para ver todos los datos en un plot bidimensional. Referencia al artículo de Brandon Panos en física solar usando PCA y t-SNE para visualizar datos multicanal (artículo).
   - Clustering: Probar aquellos métodos de clustering que tengan más relación con la distribución de los datos.

Added articles for anomaly detection and solar flares analysis

1. **Añadido el artículo "Anomalous temporal behaviour of broadband Lyα observations during solar flares from SDO/EVE":**

   - Este artículo proporciona información crucial sobre el comportamiento anómalo de la emisión Lyα durante erupciones solares, utilizando datos del Observatorio de Dinámica Solar (SDO).

2. **Añadido el artículo "Unsupervised Outlier Detection via Transformation Invariant Autoencoder":**

   - Presenta un método innovador para la detección no supervisada de outliers utilizando un autoencoder invariante a transformaciones, aplicable a datos complejos.

3. **Añadido el artículo "Self-Supervised Learning for Anomaly Detection With Dynamic Local Augmentation":**

   - Introduce un marco de aprendizaje auto-supervisado con augmentación local dinámica para mejorar la detección de anomalías a nivel de píxel.

4. **Añadido el artículo "A Tutorial Overview of Anomaly Detection in Hyperspectral Images":**

   - Proporciona una visión general y metodologías para la detección de anomalías en datos hiperespectrales, que pueden adaptarse a datos del SDO.

5. **Añadido el artículo "One-class classification with Gaussian processes":**

   - Explora la clasificación de una sola clase utilizando procesos gaussianos, un enfoque relevante para la detección de anomalías en datos con muchos ejemplos de comportamiento normal.

6. **Añadido el artículo "Deep Spatiotemporal Clustering: A Temporal Clustering Approach for Multi-dimensional Climate Data":**
   - Propone un nuevo algoritmo de clustering temporal para datos espacio-temporales de alta dimensionalidad, utilizando un autoencoder con capas CNN-RNN.

### Archivos Añadidos

- `anomalies_in_flares.pdf`
- `Unsupervised_Outlier_Detection_via_Transformation_Invariant_Autoencoder.pdf`
- `Self-Supervised_Learning_for_Anomaly_Detection_With_Dynamic_Local_Augmentation.pdf`
- `matteoli2010_anomalies_hiperspectral.pdf`
- `kemmler2013_OneClass_GPs.pdf`
- `Elmrabit.pdf`
- `Panos_2020_ApJ_891_17.pdf`

### Propósito

Estos artículos fueron añadidos para:

1. Ampliar la base teórica y metodológica del proyecto.
2. Proporcionar referencia y contexto para las técnicas y algoritmos de detección de anomalías que se implementarán.
3. Facilitar el desarrollo de nuevas metodologías basadas en trabajos previos y probados en el campo de la física solar y el aprendizaje automático.

### Próximos Pasos

1. Implementar las metodologías propuestas en los artículos para la detección de anomalías en datos del SDO.
2. Crear visualizaciones y gráficos en 2D y 3D para validar las técnicas.
3. Superponer los resultados del clustering sobre las imágenes solares para identificar patrones anómalos.

---

## Meeting 3 - 2024-11-01

**Participants:** Alyona Carolina Ivanova Araujo, Carlos José Díaz Baso, Juan C Guevara G.

### Topics Discussed:

1. **Preparación y visualización de datos multicanal:**

   - Confirmar que todos los canales se pueden descargar sin problemas y realizar una inspección inicial.
   - Comparar las escalas de píxeles y los valores de cuentas por segundo entre canales.

2. **Scatter Plots entre canales:**

   - Crear gráficos de dispersión entre pares de canales, como 171 vs 304 o 94 vs 335, para analizar distribuciones y relaciones entre los canales.
   - Identificar combinaciones inusuales de brillo entre canales que puedan indicar anomalías.

3. **Definición progresiva de características para anomalías:**

   - Comenzar usando dos canales como características y progresar hacia el uso de tres o más.
   - Analizar cómo se comportan las combinaciones de canales en las regiones brillantes (como un flare).

4. **Propuesta de Carlos:**

   - Ignorar momentáneamente la información espacial para enfocarse en cómo las anomalías se reflejan en las combinaciones multicanal.
   - Generar gráficos de dispersión para entender relaciones en múltiples canales a la vez.

5. **Downsizing de imágenes:**

   - Reducir el tamaño de las imágenes tomando cada 2 o 3 píxeles como estrategia para analizar imágenes completas manteniendo la similitud con la original.

6. **Flare y su contexto:**

   - Analizar el comportamiento del flare en imágenes completas versus subregiones pequeñas donde ocupa un porcentaje menor de la imagen.

7. **Separación por regiones:**

   - Evaluar si el análisis debe realizarse en conjunto (disco y limbo) o por separado.

8. **Limitaciones y experimentación con métodos actuales:**
   - Investigar cómo métodos como Meanshift responden a anomalías en regiones con gradientes bajos o nulos.
   - Considerar casos en los que scikit-learn no sea suficiente y evaluar herramientas más robustas como PyTorch.

---

### Próximos Pasos

1. **Análisis y preparación de datos:**

   - Verificar que los canales de datos se descarguen correctamente y explorar las diferencias de escala y valores entre ellos.
   - Crear gráficos de dispersión para pares de canales seleccionados (e.g., 171 vs 304, 94 vs 335).

2. **Definición de anomalías:**

   - Implementar un enfoque basado en combinaciones de múltiples canales para definir anomalías.
   - Progresar desde combinaciones de dos canales a combinaciones de tres o más.

3. **Exploración y experimentación:**

   - Probar técnicas de downsizing para imágenes completas.
   - Separar el análisis entre disco y limbo para identificar posibles diferencias.

4. **Herramientas y técnicas avanzadas:**

   - Realizar pruebas con scikit-learn en conjuntos de datos más grandes.
   - Evaluar la posibilidad de implementar algoritmos en PyTorch para manejar mayores volúmenes de datos.

5. **Colaboración y validación:**
   - Reunir retroalimentación sobre los métodos iniciales y ajustarlos en función de los resultados observados.

---

### Propósito

La discusión de esta reunión busca consolidar la estrategia para definir y detectar anomalías en datos solares multicanal, además de preparar el entorno para manejar datos a mayor escala.

---

## Meeting 4 - 2024-12-03

**Participants:** Alyona Carolina Ivanova Araujo, Carlos José Díaz Baso, Juan C Guevara G.

### Topics Discussed:

1.  \*\* Tareas realizadas ""

    - Exploracion y descarga de los datos (Servidor esta fallando.)
    - Graficcas de los canales AIA y para contexto, estudio y entender los datos, que es cada canal - https://www.thesuntoday.org/sun/wavelengths/
    - Se realizo una mascara del disco solar para trabajar con pixeles dentro de Limbo solar (A discutir).
    - Graficas 2D de dispersion entre pares de canales log y not logaritmicas.
    - Graficas 2D con los datos normalizados, donde exploraremos algunas rrlaciones interesantes.
    - Grafica 3D para una comprension mas profunda de como se relacionan los canales, (Canales de la fotosfera vs los de la Cromosfera.)
    - De la reunion anterior. 3. **Definición progresiva de características para anomalías:**

          - Comenzar usando dos canales como características y progresar hacia el uso de tres o más.
          - Analizar cómo se comportan las combinaciones de canales en las regiones brillantes (como un flare).

---

## Meeting 5 - 2024-12-14

**Participants:** Alyona Carolina Ivanova Araujo, Professor Luis Felipe Giraldo Trujillo, Carlos José Díaz Baso, Juan Camilo Guevara mez

### Topics Discussed:

1. **Detección de anomalías en imágenes solares con ruido:**

   - Evaluar si es mejor realizar preprocesamiento para reducir ruido antes o después de la detección de anomalías.
   - Se propuso el uso de un filtro gaussiano como preprocesamiento para suavizar las imágenes.
   - Acordamos explorar dos enfoques:
     - Preprocesar las imágenes antes de la detección.
     - Detectar anomalías en imágenes sin preprocesar y aplicar postprocesamiento.

2. **Técnicas de preprocesamiento:**

   - Uso de convoluciones para reducir ruido antes de la detección de anomalías.
   - Probar diferentes métodos para comparar su efectividad.

3. **Definición de features para modelos de detección de anomalías:**

   - Usar la posición relativa al centro del sol como feature adicional.
   - Considerar la intensidad del píxel como un factor relevante.
   - Investigar y evaluar otras posibles features como textura, bordes y varianza local.

4. **Organización del trabajo:**

   - Crear notebooks separados para experimentar con preprocesamiento y detección de anomalías.
   - Diseñar experimentos que permitan comparar los resultados de ambos enfoques.

---

### Tasks

#### Notebooks:

1. **Notebook 1:**

   - Explorar diferentes técnicas de convolución para preprocesar las imágenes.
   - Implementar y evaluar el filtro gaussiano.
   - Probar otros métodos si es necesario.

2. **Notebook 2:**
   - Implementar métodos de detección de anomalías en imágenes preprocesadas y sin preprocesar.
   - Comparar los resultados de ambos enfoques.

#### Features:

1. **Investigación:**

   - Revisar literatura sobre detección de anomalías en imágenes astronómicas.
   - Identificar posibles features relevantes para el modelo, como:
     - Posición relativa al centro del sol.
     - Intensidad del píxel.
     - Textura y bordes.
     - Varianza local.

2. **Evaluación de features:**

   - Analizar la relevancia de las features identificadas.
   - Seleccionar las más útiles para la detección de anomalías.

3. **Implementación:**
   - Calcular las features seleccionadas para cada píxel.
   - Incorporar las features al modelo.

#### Comparación de Resultados:

1. **Con y sin preprocesamiento:**

   - Evaluar la efectividad de la detección de anomalías en ambos escenarios.
   - Generar métricas de comparación para determinar la mejor estrategia.

2. **Análisis de features:**
   - Comparar la detección de anomalías utilizando diferentes combinaciones de features.

---

### Próximos Pasos

1. Crear y configurar los notebooks para preprocesamiento y detección de anomalías.
2. Investigar y documentar posibles features adicionales.
3. Diseñar experimentos para evaluar y comparar enfoques.
4. Implementar y probar las técnicas discutidas en los datos disponibles.
5. Analizar resultados y ajustar los métodos según los hallazgos.

---

### Propósito

Consolidar una estrategia robusta para la detección de anomalías en imágenes solares, explorando tanto el impacto del preprocesamiento como el diseño de features relevantes, con el fin de obtener resultados precisos y fiables.
