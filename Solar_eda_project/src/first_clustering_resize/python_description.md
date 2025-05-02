## Descripción del Script `analysis.py`: Detección de Anomalías SDO/AIA usando Isolation Forest (Versión con Umbrales Múltiples y Guardado de Figuras)

Este script realiza la detección de anomalías en imágenes solares del Solar Dynamics Observatory (SDO)/Atmospheric Imaging Assembly (AIA) utilizando el algoritmo de aprendizaje automático Isolation Forest. Procesa imágenes en el rango del Ultravioleta Extremo (EUV) de múltiples canales de AIA para identificar eventos solares inusuales o interesantes.

**Pasos Detallados (Versión Mejorada):**

1.  **Carga de Datos:** El script comienza identificando y cargando imágenes FITS de nivel 1 de SDO/AIA para un conjunto de canales EUV especificados (actualmente 94, 131, 171, 193, 211, 304 y 335 Ångstroms, excluyendo 1600 y 1700 Å). Utiliza la biblioteca `sunpy.map` para leer los archivos FITS y extraer los datos de la imagen y los metadatos. Para cada canal, carga solo la primera observación disponible de los datos descargados.

2.  **Preprocesamiento:** Para cada imagen de canal, se aplica un proceso de preprocesamiento para aislar el disco solar y preparar los datos para la detección de anomalías:
    *   **Máscara Circular:** Se crea una máscara circular basada en el radio solar (metadato `rsun_obs`) para aislar el disco solar y eliminar el espacio de fondo de la imagen.
    *   **Redimensionamiento:** Tanto los datos de la imagen original como la máscara circular se redimensionan a un tamaño consistente de 512x512 píxeles para asegurar la uniformidad entre canales.
    *   **Aplicación de la Máscara:** Se aplica la máscara circular a la imagen redimensionada. Los píxeles fuera del disco solar, considerados como fondo, se establecen en `NaN` (No es un Número).

3.  **Preparación de Datos para Detección de Anomalías:** Las imágenes preprocesadas de todos los canales AIA seleccionados se preparan para ser introducidas al algoritmo Isolation Forest:
    *   **Concatenación de Canales:** Las imágenes enmascaradas de cada canal se apilan juntas a lo largo de una nueva dimensión "canal", creando un array 3D.
    *   **Remodelación a Formato Píxel-Característica:** Este array 3D se remodela en un array 2D. En este formato, cada fila representa un solo píxel de la imagen solar, y las columnas representan los valores de intensidad de ese píxel en los diferentes canales de AIA. Esta representación píxel-característica es adecuada para Isolation Forest.
    *   **Eliminación de NaN:** Se eliminan del conjunto de datos remodelado las filas (píxeles) que contienen valores `NaN` (píxeles fuera del disco solar), ya que Isolation Forest no puede manejar valores `NaN`.
    *   **Escalado Robusto:** Los datos limpios se escalan utilizando `RobustScaler` de scikit-learn. El escalado robusto se utiliza para reducir el impacto de posibles valores atípicos en los datos y para asegurar que todos los canales contribuyan por igual al proceso de detección de anomalías, independientemente de sus escalas de intensidad.

4.  **Detección de Anomalías con Isolation Forest:** El paso central de detección de anomalías se realiza utilizando el algoritmo Isolation Forest:
    *   **Inicialización y Entrenamiento del Modelo:** Se inicializa un modelo `IsolationForest` de scikit-learn con un parámetro `contamination` especificado (una estimación de la proporción de anomalías en los datos) y un `random_state` para la reproducibilidad. Luego, el modelo se entrena (se ajusta) con los datos de píxeles preparados y escalados.
    *   **Cálculo de Puntuaciones de Anomalía:** Se utiliza el método `decision_function` del modelo Isolation Forest entrenado para calcular las puntuaciones de anomalía para cada píxel. La `decision_function` proporciona puntuaciones de anomalía continuas, donde las puntuaciones más altas indican píxeles "normales" y las puntuaciones más bajas (más negativas) indican píxeles más "anómalos".

5.  **Visualización y Guardado de Anomalías (para múltiples umbrales):**  Esta versión mejorada itera sobre una lista de umbrales de anomalía y, para cada umbral:
    *   **Creación de Mapa de Anomalías:** Para cada canal, se crea un mapa de anomalías remodelando las puntuaciones de anomalía de nuevo a la cuadrícula de imagen de 512x512 y estableciendo un umbral. Los píxeles con puntuaciones de anomalía por debajo del `anomaly_threshold` se consideran anomalías.
    *   **Superposición en la Imagen Original:** Para cada canal, se genera un subplot. Muestra la imagen original enmascarada para ese canal, superpuesta con una máscara de anomalías de color rojo-amarillo. La máscara de anomalías es semitransparente para permitir que se vea la imagen solar subyacente.
    *   **Guardado de Figuras:** En lugar de mostrar las figuras en pantalla, el script ahora guarda cada figura como un archivo PNG en un directorio de salida especificado por el usuario (o `./output_figures` por defecto). El nombre del archivo incluye el valor del umbral de anomalía utilizado.
    *   **Uso de Argumentos de Línea de Comandos:** La funcionalidad de umbrales múltiples y directorio de salida se implementa utilizando la biblioteca `argparse`. Esto permite al usuario especificar los umbrales de anomalía deseados y el directorio de salida al ejecutar el script desde la línea de comandos.

**Nuevas Funcionalidades Introducidas en esta Versión:**

*   **Umbrales de Anomalía Múltiples:** El script ahora puede procesar y generar visualizaciones para múltiples umbrales de anomalía en una sola ejecución. Los umbrales se especifican como argumentos de línea de comandos.
*   **Guardado Automático de Figuras:** En lugar de mostrar las figuras interactivamente, el script guarda automáticamente cada figura en un archivo PNG en un directorio de salida especificado. Esto facilita la comparación de resultados con diferentes umbrales y el procesamiento por lotes.
*   **Argumentos de Línea de Comandos con `argparse`:** Se utiliza la biblioteca `argparse` para manejar los argumentos de línea de comandos, incluyendo la especificación de múltiples umbrales de anomalía (`--anomaly_thresholds`) y el directorio de salida (`--output_dir`). Esto hace que el script sea más flexible y fácil de usar desde la línea de comandos.

**Cómo Ejecutar el Script con Múltiples Umbrales y Guardar Figuras:**

Para ejecutar el script con múltiples umbrales de anomalía y guardar las figuras, utilice la línea de comandos de la siguiente manera:

```bash
python analysis.py --anomaly_thresholds -0.2 -0.1 0.0 0.1 --output_dir ./mis_figuras_anomalias