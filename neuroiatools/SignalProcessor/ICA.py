import mne
from neuroiatools.DisplayData.plotEEG import plotEEG
from mne.preprocessing import ICA

def getICA(data,n_components=None, method='fastica', random_state=None, max_iter=1000,reject_by_annotation=True, picks=None, **kwargs):
    """
    Apply Independent Component Analysis (ICA) to EEG data to identify and remove artifacts.

    Parameters
    ----------
    data : mne.io.RawArray
        Datos EEG en forma de objeto mne.io.RawArray.
    sfreq : float
        Frecuencia de muestreo de los datos EEG en Hz.
    n_components : int|None
        Número de componentes a extraer. Si es None, se extraen todos los componentes. Por defecto es None.
    method : str
        Método de ICA a utilizar. Puede ser 'fastica' o 'infomax'. Por defecto es 'fastica'.
    random_state : int|None
        Estado aleatorio para reproducibilidad. Por defecto es None.
    max_iter : int
        Número máximo de iteraciones. Por defecto es 1000.
    reject_by_annotation : bool, optional
        Whether to reject annotated bad segments before applying ICA. Default is True.
    picks : str, list, o None
        Canales a seleccionar. Si es None, se seleccionan todos los canales. Por defecto es None.
    **kwargs : dict
        Argumentos adicionales para el método ICA.

    Retorna
    -------
    ica : mne.preprocessing.ICA
        Objeto ICA entrenado.

    """
    ##validamos que rawdata sea un objeto mne.io.RawArray
    if not isinstance(data, mne.io.RawArray):
        raise ValueError("data debe ser un objeto mne.io.RawArray.")

    ## Creamos un objeto ICA
    ica = ICA(n_components=n_components, method=method, random_state=random_state, max_iter=max_iter)

    ## Entrenamos el ICA
    ica.fit(data, reject_by_annotation=reject_by_annotation, picks=picks, **kwargs)

    return ica

if __name__ == "__main__":
    pass
