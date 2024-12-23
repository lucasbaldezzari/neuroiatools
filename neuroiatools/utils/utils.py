import importlib.resources as pkg_resources
from neuroiatools import datasets

def load_file(filename):
    """
    Carga un archivo desde el directorio 'erders'.
    
    Parameters:
        filename (str): Nombre del archivo a cargar.

    Returns:
        str: Contenido del archivo.
    """
    if filename not in pkg_resources.contents(datasets):
        raise FileNotFoundError(f"El archivo {filename} no se encontró en 'erders'.")
    
    with pkg_resources.open_text(datasets, filename) as f:
        return f.read()