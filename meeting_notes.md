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
