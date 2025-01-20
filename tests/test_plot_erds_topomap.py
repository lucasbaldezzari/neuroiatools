from neuroiatools.SignalProcessor import Filter
from neuroiatools.SignalProcessor.tfr import compute_tfr, plot_ERDS_topomap
from neuroiatools.EEGManager.RawArray import makeRawData
import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

##cargo datasets/raweeg_executed_tasks.hdf5
raweeg = h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")["raw_eeg"][:63,:] ##no usamos el último canal dado que es EMG
eventos = pd.read_csv("datasets\\events_executed_tasks.txt")
data =  h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")
sfreq = 512

# Tiempos de los eventos en segundos
event_times = np.astype(eventos["event_time"].values,int)
event_labels = eventos["class_name"].values

##filtramos 
filtro = Filter.Filter(lowcut=1, highcut=36, notch_freq=50.0, notch_width=2.0, sample_rate=512.0)
filtered_eeg = filtro.filter_data(raweeg)

# Configuración de parámetros para computar la TFR
tmin = -3
tmax = 5
dt = 0.5
fmin = 5     # Frecuencia mínima de interés
fmax = 36    # Frecuencia máxima de interés
 
##elecrtodos a usar
##***IMPORTANTE***
##Los nombres de los electrodos deben coincidir con el montaje de la gorra de gtec. LOS NOMBRES AQUÍ USADOS NO SE CORRESPONDEN CON LOS USADOS EN EL REGISTRO DE DATOS
##SE DEBE CREAR UN MONTAJE CON g.MONTAGECREATOR
ch_names = ['FP1', 'FPz', 'FP2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'T10', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
            'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']

###Creación de un Montage para el posicionamiento de los electrodos
montage = mne.channels.read_custom_montage("tests\\montage.sfp")
montage.plot(show_names=True)

##creamos el objeto RawArray
rawdata = makeRawData(filtered_eeg, sfreq, channel_names=ch_names, montage=montage)
rawdata.crop(tmin=33)

##computo la TFR
tfr, freqs, times = compute_tfr(
    rawdata, event_times, event_labels, tmin-dt, tmax+dt, fmin, fmax, n_cycles=20,
    reject=None, baseline=(-3,-1), baseline_mode="percent", baseline_cropping=(tmin,tmax) )

##seteo el montaje
tfr.info.set_montage(montage)
print(tfr.info['dig'])

##grafico los mapas topográficos
times=[-0.5, 0, 0.5,1] ##tiempos en segundos
class_interest = "IZQUIERDA"

plot_ERDS_topomap(
    tfr=tfr,
    times=times,
    class_interest=class_interest,
    bands_interest=["alpha", "beta"],
    freq_bounds={"alpha": (8, 13), "beta": (14, 30)},
    title=f"Mapas Topográficos ERDS - Mano: {class_interest}",
    show=True,
    save=True,filename="tests//erds_topomap.png",
    apply_cnorm=True,
    vmin=-1.5,
    vmax=1,
    cmap="PiYG",
    colorbar=True,
    show_names=False,
    figsize=(16, 12),
    )