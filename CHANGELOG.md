# Changelog SOLAR

Solar Observer Learning Anomaly Recognition (SOLAR)

All notable changes to this project will be documented in this file.

This format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial setup and structure of the repository.
- GitHub Actions workflows for CI/CD.
- Pre-commit hooks for code quality.

### Changed

- Initial discussions and planning for the project.
- Defined initial strategy and objectives.

## [0.1.0] - 2024-07-02

### Added

- Initial project launch with basic setup and documentation.
- Created README.md with project description and goals.

## Meeting Notes

### Meeting 1 - 2024-06-05

**Participants:** Alyona Carolina Ivanova Araujo, Professor Luis Felipe Giraldo Trujillo, Carlos José Díaz Baso, Juan Camilo Guevara mez [Emails hidden for privacy]

#### Topics Discussed:

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

## Contributors

- Alyona Carolina Ivanova Araujo
