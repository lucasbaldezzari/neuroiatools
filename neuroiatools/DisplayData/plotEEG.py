import numpy as np
import mne

def plotEEG(eeg_data, title = "EEG", scalings=dict(eeg=40e-6),
            color = {"eeg":"blue"}, bgcolor = "#eaeded", **kwargs):
    """
    Genera una gráfica interactiva para analizar un registro de EEG con eventos marcados.

    Utiliza la librería MNE para generar la gráfica interactiva.

    Parámetros:
    - eeg_data (mne.io.RawArray): Datos de EEG en forma de objeto mne.io.RawArray.
    - sfreq (float): Frecuencia de muestreo en Hz.
    - title (str, opcional): Título de la gráfica.
    - scalings (dict): Escalas para los canales. Por defecto, 'eeg'=40e-6. Puede ser "auto" o bien "None".
    - color (dict): Especifíca el color de los trazos del EEG. Por defecto es azul.
    - bgcolor: Color del fondo.
    - kwargs: Argumentos adicionales para la función `mne.io.Raw.plot`. Ver en https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot

    Retorna:
    - None. Muestra una gráfica interactiva.
    """
    ##validamos que eeg_data sea un objeto mne.io.RawArray
    if not isinstance(eeg_data, mne.io.RawArray):
        raise ValueError("eeg_data debe ser un objeto mne.io.RawArray.")
    
    ##Graficamos
    eeg_data.plot(scalings=scalings,
                  title=title,
                  color = color, bgcolor = bgcolor,
                  **kwargs)

if __name__ == "__main__":

    from neuroiatools.EEGManager.RawArray import makeRawData

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

    eeg_data = makeRawData(eeg_data, sfreq, event_times, event_labels)

    # Llamar a la función
    plotEEG(eeg_data,scalings = 100e-6,show=True, block=True,duration = 20, start = 0)