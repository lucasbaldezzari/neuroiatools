import numpy as np
import mne

def plotEEG(eeg_data, sfreq, event_times, event_labels, channel_names=None,
            title = "EEG", scalings=dict(eeg=40e-6), color = {"eeg":"blue"}, bgcolor = "#eaeded", **kwargs):
    """
    Genera una gráfica interactiva para analizar un registro de EEG con eventos marcados.

    Utiliza la librería MNE para generar la gráfica interactiva.

    Parámetros:
    - eeg_data (np.ndarray): Array de forma [canales, muestras] con los datos de EEG.
    - sfreq (float): Frecuencia de muestreo en Hz.
    - event_times (list or np.ndarray): Momentos en segundos donde ocurrieron los eventos.
    - event_labels (list): Nombres de los eventos, deben coincidir en longitud con `event_times`.
    - channel_names (list, opcional): Nombres de los canales. Si no se proporcionan, se generarán automáticamente.
    - title (str, opcional): Título de la gráfica.
    - scalings (dict): Escalas para los canales. Por defecto, 'eeg'=40e-6. Puede ser "auto" o bien "None".
    - color (dict): Especifíca el color de los trazos del EEG. Por defecto es azul.
    - bgcolor: Color del fondo.
    - kwargs: Argumentos adicionales para la función `mne.io.Raw.plot`. Ver en https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot

    Retorna:
    - None. Muestra una gráfica interactiva.
    """
    if eeg_data.ndim != 2:
        raise ValueError("eeg_data debe ser un array de forma [canales, muestras].")

    if len(event_times) != len(event_labels):
        raise ValueError("La cantidad de event_times debe coincidir con la cantidad de event_labels.")

    n_channels, n_samples = eeg_data.shape
    duration = n_samples / sfreq  # Duración del registro en segundos.

    ## Generamos nombres de canales si no se proporcionaron
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    ## Creamos un objeto mne.Info con la información de los canales
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')

    ## Creamos un objeto mne.RawArray con los datos de EEG
    raw = mne.io.RawArray(eeg_data, info)

    ## Creamos un objeto mne.Annotations con los eventos
    annotations = mne.Annotations(onset=event_times,  ## Eventos en segundos
                                   duration=[0] * len(event_times), 
                                   description=event_labels)
    
    raw.set_annotations(annotations)
    
    ##Graficamos
    raw.plot(scalings=scalings,
             title=title,
             color = color, bgcolor = bgcolor,
             **kwargs)

if __name__ == "__main__":

    # Ejemplo con datos ficticios
    n_channels = 64
    sfreq = 512  # Frecuencia de muestreo en Hz
    duration_sec = 900  # Duración de 5 minutos
    n_samples = int(sfreq * duration_sec)

    # Generar datos ficticios (ruido aleatorio)
    np.random.seed(42)
    eeg_data = np.random.randn(n_channels, n_samples)*20e-6

    # Eventos ficticios
    event_times = [10, 50, 120, 200, 800]  # En segundos
    event_labels = ['Inicio', 'Tarea 1', 'Tarea 2', 'Pausa', 'Fin']

    # Llamar a la función
    plotEEG(eeg_data, sfreq, event_times, event_labels, scalings = 100e-6,show=True, block=True,
            duration = 20, start = 0)