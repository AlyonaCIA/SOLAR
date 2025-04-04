# Installation Instructions / Instrucciones de Instalación

## 1. Clone the Repository / Clonar el Repositorio
Clone this repository to your local machine:

```bash
git clone <https://github.com/AlyonaCIA/SOLAR.git>
cd <SOLAR>
```

## 2. Create and Activate a Virtual Environment / Crear y Activar un Entorno Virtual
Create a virtual environment to isolate project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux / En macOS/Linux
```

On Windows, use:

```powershell
.venv\Scripts\activate
```

## 3. Install Dependencies / Instalar las Dependencias
Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 4. Update SSL Certificates (macOS Only) / Actualizar Certificados SSL (Solo en macOS)
If you are using macOS, make sure to update the SSL certificates to avoid connection errors:

```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

Replace `3.x` with your Python version.

Reemplaza `3.x` con la versión de Python que estás utilizando.

## 5. Run the Download Script / Ejecutar el Script de Descarga
Run the `download_sdo.py` script to download the SDO images:

```bash
python tmp/download_sdo.py
```

The downloaded images will be stored in the `sdo_data` directory, with subdirectories for each channel (e.g., `dia_1600` and `dia_1700`).

Las imágenes descargadas se almacenarán en el directorio `sdo_data`, con subdirectorios específicos para cada canal (por ejemplo, `dia_1600` y `dia_1700`).

## 6. Verify Downloads / Verificar las Descargas
You can verify the downloaded files by navigating to the `sdo_data` directory:

```bash
ls ./sdo_data/
```

