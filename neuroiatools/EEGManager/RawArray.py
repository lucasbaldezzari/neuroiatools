import mne

def makeRawData(data, sfreq, event_times=None, event_labels=None, channel_names=None, montage=None):
    """
    Función para crear un objeto mne.io.RawArray a partir de un arreglo de datos EEG.

    Parámetros
    ----------
    data : np.ndarray
        Arreglo de datos EEG con forma (n_canales, n_muestras).
    sfreq : float
        Frecuencia de muestreo de los datos EEG en Hz.
    event_times : list or np.ndarray, opcional
        Tiempo de los eventos en segundos.
    event_labels : list, opcional
        Etiquetas de los eventos.
    channel_names : list or None, opcional
        Lista de nombres de canales. Si es None, los nombres se generan automáticamente. Por defecto es None.
    montage : mne.channels.Montage or None, opcional

    Retorna
    -------
    rawdata : mne.io.RawArray
    """
    ##validamos dimensión
    if data.ndim != 2:
        raise ValueError("Los datos de entrada deben tener la forma (n_channels, n_samples).")
    
    ##Nombre de los canales
    if channel_names is None:
        info = mne.create_info(ch_names=[f'EEG {i+1}' for i in range(data.shape[0])], sfreq=sfreq, ch_types='eeg')
    else:
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')

    ##montaje de electrodos
    if montage is not None:
        info.set_montage(montage)

    ## Creamos un objeto mne.RawArray con los datos de EEG
    rawdata = mne.io.RawArray(data, info)

    ## si event_times y event_labels no son None, creamos un objeto mne.Annotations
    if event_times is not None and event_labels is not None:
        ## Creamos un objeto mne.Annotations con los eventos
        annotations = mne.Annotations(onset=event_times,  ## Eventos en segundos
                                    duration=[0] * len(event_times), 
                                    description=event_labels)

        rawdata.set_annotations(annotations)

    return rawdata ##retornamos el objeto mne.io.RawArray

if __name__ == "__main__":
    import h5py
    import numpy as np
    import pandas as pd
    from neuroiatools.SignalProcessor import Filter

    ##cargo datasets/raweeg_executed_tasks.hdf5
    raweeg = h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")["raw_eeg"][:63,:] ##no usamos el último canal dado que es EMG
    eventos = pd.read_csv("datasets\\events_executed_tasks.txt")

    sfreq = 512

    # Tiempos de los eventos en segundos
    event_times = np.astype(eventos["event_time"].values,int)/sfreq ##marcado de eventos en segundos
    event_labels = eventos["class_name"].values

    ##filtramos 
    filtro = Filter.Filter(lowcut=1, highcut=36, notch_freq=50.0, notch_width=2.0, sample_rate=sfreq)
    filtered_eeg = filtro.filter_data(raweeg)

    ##creamos el objeto RawArray
    rawdata = makeRawData(filtered_eeg, sfreq, event_times, event_labels)

    print(rawdata.ch_names)
