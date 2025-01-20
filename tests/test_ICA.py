from neuroiatools.SignalProcessor import Filter
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
event_times = np.astype(eventos["event_time"].values,int)/sfreq ##marcado de eventos en segundos
event_labels = eventos["class_name"].values

##filtramos 
filtro = Filter.Filter(lowcut=1, highcut=36, notch_freq=50.0, notch_width=2.0, sample_rate=sfreq)
filtered_eeg = filtro.filter_data(raweeg)

ch_names = ['FP1', 'FPz', 'FP2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'T10', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
            'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']

###Creación de un Montage para el posicionamiento de los electrodos
montage = mne.channels.read_custom_montage("tests\\montage.sfp")

##creamos el objeto RawArray
eeg_data = makeRawData(filtered_eeg, sfreq, channel_names=ch_names, montage=montage, event_times=event_times, event_labels=event_labels)
eeg_data.crop(tmin=33)

ica = getICA(eeg_data, n_components = 30)
ica.plot_sources(eeg_data)
ica.plot_components()

# ica.plot_overlay(eeg_data, exclude=[20], picks="eeg")

ica.plot_properties(eeg_data, picks=[0,4,21,22,23], psd_args={'fmax': 35.}, image_args={'sigma': 1.})

ica.exclude = [3,21,22]
eeg_data_reconstructed = eeg_data.copy()
eeg_data_reconstructed = ica.apply(eeg_data_reconstructed)

plotEEG(eeg_data,scalings = 40,show=True, block=True,
    duration = 30, remove_dc = True, bad_color = "red")

plotEEG(eeg_data_reconstructed,scalings = 40,show=True, block=True,
    duration = 30, remove_dc = True, bad_color = "red")