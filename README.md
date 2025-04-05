# Installation Instructions / Instrucciones de Instalación

Tareas pendientes 
1. Hacer git push al dvc
2. Asignar permisos a los u

1) Iniciar con el git clone del proyecto 
2) Inicializar git y dvc 


## MAC OS

En la terminal : 

0) Descargar  CLI de google 
curl -O "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-x86_64.tar.gz?hl=es-419"

1) Descomprimir 
tar -xvzf google-cloud-cli-darwin-x86_64.tar.gz 

2) Navegar a la ubicación del archivo 
cd google-cloud-sdk

3) Instalar 
./install.sh

4) Actualización del Archivo .zshrc

5) Iniciar google CLI
gcloud init

6) Agregar credenciales personales , el correo del proyecto es "solarproyect2024@gmail.com".

7) Agregar el bucket a dvc 
dvc remote add -d gcp-remote gs://ml-project-dvc-bucket

8) Usar la llave de acceso 
dvc remote modify gcp-remote --local gdrive_service_account_json_file gcp-key.json

Se debe actualizar el archivo de requermientos 
## UBUNTU
sudo apt install google-cloud-sdk

# Pasos para clonar el repositorio y acceder a los datos 

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


# 📦 Proyecto SOLAR - Acceso a Datos con DVC y GCP / SOLAR Project - Accessing Data with DVC and GCP

Este documento describe los pasos necesarios para clonar este repositorio, instalar sus dependencias y acceder a los datos versionados usando DVC con almacenamiento en Google Cloud Platform (GCP).

This document describes the steps needed to clone this repository, install its dependencies, and access versioned data using DVC with storage in Google Cloud Platform (GCP).

---

## 1. Clonar el Repositorio / Clone the Repository

```bash
git clone https://github.com/AlyonaCIA/SOLAR.git
cd SOLAR
```

---

## 2. Crear y Activar un Entorno Virtual / Create and Activate a Virtual Environment

### En macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### On Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Instalar las Dependencias / Install the Dependencies

Instala las dependencias del proyecto, incluyendo soporte para DVC con Google Cloud:

Install the project dependencies, including DVC with Google Cloud support:

```bash
pip install -r requirements.txt
pip install "dvc[gcs]"
```

---

## 4. Configurar el Acceso a GCP / Configure Access to GCP

### 🔐 Cuenta de Servicio / Service Account (Recomendado / Recommended)

El acceso a los datos está restringido al uso exclusivo de una cuenta de servicio.  
Access to the data is restricted to using a service account only.

1. Obtén el archivo `.json` con las credenciales de la cuenta de servicio.  
   Get the `.json` file with service account credentials.

2. Exporta la variable de entorno / Export the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/ruta/a/tu/credenciales.json"
```

Asegúrate de que tus compañeros tengan el archivo `.json` y que su cuenta de servicio tenga al menos el rol:  
Make sure your teammates have the `.json` file and that the service account has at least the role:  

- `roles/storage.objectViewer`

> ✅ No es necesario usar `gcloud auth`, ya que el acceso se realiza exclusivamente mediante la cuenta de servicio.  
> ✅ `gcloud auth` is not necessary, since access is handled exclusively via the service account.

---

## 5. Descargar los Datos con DVC / Download Data with DVC

Una vez autenticado y con el entorno activado, ejecuta:  
Once authenticated and with the virtual environment activated, run:

```bash
dvc pull
```

Esto descargará todos los archivos de datos versionados desde el bucket de GCP a sus ubicaciones originales dentro del proyecto.  
This will download all versioned data files from the GCP bucket to their original locations in the project.

---

## 6. Verificar los Archivos Descargados / Verify Downloaded Files

Puedes revisar el contenido de los directorios de datos, por ejemplo:  
You can check the contents of the data directories, for example:

```bash
ls ./sdo_data/
```

Esto debería mostrar subdirectorios como `dia_1600` y `dia_1700`, dependiendo de cómo esté estructurado el dataset.  
This should show subdirectories like `dia_1600` and `dia_1700`, depending on the dataset structure.

---

## 💡 Notas Adicionales / Additional Notes

- **Nunca subas archivos de credenciales `.json` al repositorio.** Añádelos a tu `.gitignore`.  
  **Never upload `.json` credential files to the repository.** Add them to your `.gitignore`.

- Asegúrate de que el bucket GCS esté configurado correctamente con los permisos adecuados para lectura.  
  Ensure that the GCS bucket is properly configured with appropriate read permissions.

- Puedes añadir más remotos o cambiar el remoto por defecto usando `dvc remote`.  
  You can add more remotes or change the default remote using `dvc remote`.


