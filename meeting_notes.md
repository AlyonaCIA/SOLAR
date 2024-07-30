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

## Meeting 2 - 2024-07-10

**Participants:** Alyona Carolina Ivanova Araujo, Professor Luis Felipe Giraldo Trujillo, Carlos José Díaz Baso, Juan Camilo Guevara mez

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
