import importlib.resources as resources
from neuroiatools import datasets
import pandas as pd
import h5py
import numpy as np

def load_file(filename, file_type=None):
    """
    Carga un archivo desde el directorio 'datasets' en diferentes formatos.

    Parameters:
        filename (str): Nombre del archivo a cargar.
        file_type (str, optional): Tipo de archivo. Opcional si puede ser inferido del nombre.
            Valores soportados: 'txt', 'npy', 'hdf5'.

    Returns:
        object: Contenido del archivo (str, np.ndarray, dict, etc.).
    """
    # Inferir el tipo de archivo si no se especifica
    if file_type is None:
        file_type = filename.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':##abro como pandas
            with resources.path(datasets, filename) as path:
                return pd.read_csv(path)
        elif file_type == 'npy':
            with resources.path(datasets, filename) as path:
                return np.load(path, allow_pickle=True)
        elif file_type in ('hdf5', 'h5'):
            with resources.path(datasets, filename) as path:
                with h5py.File(path, 'r') as f:
                    return {key: f[key][()] for key in f.keys()}
        else:
            raise ValueError(f"Tipo de archivo no soportado: {file_type}")
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {filename} no se encontró en 'datasets'.")