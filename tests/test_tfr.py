from neuroiatools.SignalProcessor import Filter
from neuroiatools.SignalProcessor.tfr import compute_tfr, plotTFRERDS, plotERDSLines
import h5py
import numpy as np
import pandas as pd

##cargo datasets/raweeg_executed_tasks.hdf5
raweeg = h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")["raw_eeg"][:63,:] ##no usamos el último canal dado que es EMG
eventos = pd.read_csv("datasets\\events_executed_tasks.txt")

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

channels=["28","36"]

tfr, freqs, times = compute_tfr(
    filtered_eeg, sfreq, event_times, event_labels, tmin-dt, tmax+dt, fmin, fmax, n_cycles=20,
    pick_channels=channels, reject=dict(eeg=60), baseline=(-3,-1), baseline_mode="percent", baseline_cropping=(tmin,tmax) )

files_names = [f"tests\\tfr_chan{ch}.png" for ch in channels]
plotTFRERDS(tfr,event_ids=dict(DERECHA=0, IZQUIERDA=1), ch_names=channels, vmin=None, vmax=None,
            show=True, save=False, files_names=files_names, dpi=300, figsize=(16, 6))

plotERDSLines(tfr, channels_order=channels, bands_interest=["alpha", "beta"], title=None,
                freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}, figsize=(16, 6),
                color_palette="blend:#8e44ad,#3498db", n_colors=2,show=True,save=False, filename="tests\\erdslines.png", dpi=300)