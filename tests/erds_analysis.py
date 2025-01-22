from neuroiatools.SignalProcessor import Filter
from neuroiatools.SignalProcessor.tfr import compute_tfr, plotTFRERDS, plotERDSLines
from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
import h5py
import numpy as np
import pandas as pd
import mne


##cargo datasets/raweeg_executed_tasks.hdf5
raweeg = h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")["raw_eeg"][:63,:] ##no usamos el último canal dado que es EMG
eventos = pd.read_csv("datasets\\events_executed_tasks.txt")

sfreq = 512

# Tiempos de los eventos en segundos
event_times_samples = np.astype(eventos["event_time"].values,int)
event_times_seconds = event_times_samples/sfreq
event_labels = eventos["class_name"].values

###********** FILTRANDO DATOS **********###
filtro = Filter.Filter(lowcut=1, highcut=40, notch_freq=50.0, notch_width=2.0, sample_rate=sfreq)
filtered_eeg = filtro.filter_data(raweeg)

###********** OBJETO RawArray **********###
##creamos el objeto RawArray
rawdata = makeRawData(filtered_eeg, sfreq)
rawdata.crop(tmin=33)

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
# montage.plot(show_names=True)

##creamos el objeto RawArray
eeg_data = makeRawData(filtered_eeg, sfreq, channel_names=ch_names, montage=montage, event_times=event_times_seconds, event_labels=event_labels)
eeg_data.crop(tmin=33)

###********** APLICANDO ICA **********###

ica = getICA(eeg_data, n_components = 30)
ica.plot_sources(eeg_data)
ica.plot_components()

# ica.plot_overlay(eeg_data, exclude=[20], picks="eeg")

ica.plot_properties(eeg_data, picks=[0,4,21,22,23], psd_args={'fmax': 35.}, image_args={'sigma': 1.})

ica.exclude = [3,21]
eeg_data_reconstructed = eeg_data.copy()
eeg_data_reconstructed = ica.apply(eeg_data_reconstructed)

##señal antes de aplicar ICA
plotEEG(eeg_data,scalings = 40,show=True, block=True, start = 62,
    duration = 30, remove_dc = True, bad_color = "red")
##señal después de aplicar ICA
plotEEG(eeg_data_reconstructed,scalings = 40,show=True, block=True, start = 62,
    duration = 30, remove_dc = True, bad_color = "red")

###********** TIME FREQUENCY ANALYSIS **********###

# Configuración de parámetros para computar la TFR
tmin = -3
tmax = 5
dt = 0.5
fmin = 5     # Frecuencia mínima de interés
fmax = 36    # Frecuencia máxima de interés

channels = ["C3", "C4"]

tfr, freqs, times = compute_tfr(
    eeg_data_reconstructed, event_times_samples, event_labels, tmin-dt, tmax+dt, fmin, fmax, n_cycles=20,
    pick_channels=channels, reject=dict(eeg=100), baseline=(-3,-1), baseline_mode="percent", baseline_cropping=(tmin,tmax) )

files_names = [f"tests\\figures\\tfr_chan{ch}.png" for ch in channels]
plotTFRERDS(tfr,event_ids=dict(DERECHA=0, IZQUIERDA=1), ch_names=channels, vmin=None, vmax=None,
            show=True, save=False, files_names=files_names, dpi=300, figsize=(16, 6))

plotERDSLines(tfr, channels_order=channels, bands_interest=["alpha", "beta"], title=None,
                freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}, figsize=(16, 6),
                color_palette="blend:#8e44ad,#3498db", n_colors=2,show=True,save=False, filename="tests\\figures\\erdslines.png", dpi=300)