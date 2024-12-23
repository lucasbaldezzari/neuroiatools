import requests
from pathlib import Path

def download_data(file_name, save_dir="datasets", repo_url="https://github.com/lucasbaldezzari/neuroiatools/tree/main/neuroiatools/datasets"):
    """
    Descarga un archivo de datos desde un repositorio de GitHub y lo guarda localmente.
    
    Parameters:
        file_name (str): Nombre del archivo que deseas descargar (e.g., "data1.npy").
        save_dir (str): Directorio local donde se guardará el archivo.
        repo_url (str): URL base del repositorio donde están los datos.
        
    Returns:
        str: Ruta completa del archivo descargado.
    """
    # Crear el directorio si no existe
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # URL completa del archivo
    file_url = f"{repo_url}/{file_name}"
    local_file_path = save_path / file_name
    
    # Descargar solo si no existe localmente
    if not local_file_path.exists():
        print(f"Descargando {file_name} desde {file_url}...")
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(local_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Archivo guardado en {local_file_path}")
        else:
            raise FileNotFoundError(f"No se pudo descargar el archivo: {file_url}. Status code: {response.status_code}")
    else:
        print(f"El archivo {file_name} ya existe en {local_file_path}.")
    
    return str(local_file_path)