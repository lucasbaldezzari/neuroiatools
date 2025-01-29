from neuroiatools.SignalProcessor import Filter
from neuroiatools.SignalProcessor.tfr import compute_tfr, plotTFRERDS, plotERDSLines, plot_ERDS_topomap
from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
import h5py
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


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
            'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
            'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']

###Creación de un Montage para el posicionamiento de los electrodos
montage = mne.channels.read_custom_montage("tests\\montage.sfp")
# montage.plot(show_names=True)

##creamos el objeto RawArray
eeg_data = makeRawData(filtered_eeg, sfreq, channel_names=ch_names, montage=montage, event_times=event_times_seconds, event_labels=event_labels)
eeg_data.crop(tmin=33)

eeg_data.info["bads"] = ["C3","Cz","C4"] ##para marcado de electrodos en los gráficos 2D y 3D debajo

## Graficamos los electrodos en 2D y 3D
fig = plt.figure(figsize=(16, 10))
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection="3d")
eeg_data.plot_sensors(ch_type="eeg", axes=ax2d,show_names=True)
eeg_data.plot_sensors(ch_type="eeg", axes=ax3d, kind="3d",show_names=True)
ax3d.view_init(azim=70, elev=15)

eeg_data.info["bads"] = [] ##volvemos a setear los electrodos como al inicio

###********** APLICANDO ICA **********###

ica = getICA(eeg_data, n_components = 40)

##graficamos los componentes independientes a partir de la matriz de fuentes
ica.plot_sources(eeg_data)

##graficamos los mapas topográficos de los componentes independientes a partir de la matriz de mezcla
ica.plot_components()

##graficamos las propiedades de los componentes independientes
ica.plot_properties(eeg_data, picks=[0], psd_args={'fmax': 35.}, image_args={'sigma': 1.})

comps_to_exclude = [3,22,28,26,35,36] ##componentes a excluir
##graficamos los componentes independientes superpuestos en la señal original
ica.plot_overlay(eeg_data, exclude=comps_to_exclude, picks=["C3", "C4"])

## LIMPANDO NUESTRA SEÑAL DE LOS COMPONENTES NO DESEADOS
ica.exclude = comps_to_exclude ##componentes a excluir
eeg_data_reconstructed = eeg_data.copy() ##copia de los datos originales
eeg_data_reconstructed = ica.apply(eeg_data_reconstructed) ##aplicamos ICA

##señal antes de aplicar ICA
plotEEG(eeg_data,scalings = 40,show=True, block=True, start = 62,
    duration = 30, remove_dc = True, bad_color = "red")
##señal después de aplicar ICA
plotEEG(eeg_data_reconstructed,scalings = 40,show=True, block=True, start = 62,
    duration = 30, remove_dc = True, bad_color = "red")

##elimino canal AF8
eeg_data_reconstructed.drop_channels(["AF8"])

###********** TIME FREQUENCY ANALYSIS **********###

# Configuración de parámetros para computar la TFR
tmin = -2
tmax = 4
dt = 0.5
fmin = 5     # Frecuencia mínima de interés
fmax = 36    # Frecuencia máxima de interés

channels_selected = ["C3", "Cz","C4"]

tfr, freqs, times = compute_tfr(
    eeg_data_reconstructed, event_times_samples, event_labels, tmin-dt, tmax+dt, fmin, fmax, n_cycles=20,
    pick_channels=channels_selected, reject=dict(eeg=80), baseline=(-2,-1), baseline_mode="percent", baseline_cropping=(tmin,tmax) )

files_names = [f"tests\\figures\\tfr_chan{ch}.png" for ch in channels_selected]
plotTFRERDS(tfr,event_ids=dict(DERECHA=0, IZQUIERDA=1), ch_names=channels_selected, vmin=None, vmax=None,
            show=True, save=False, files_names=files_names, dpi=300, figsize=(16, 6))

plotERDSLines(tfr, channels_order=channels_selected, bands_interest=["alpha", "beta"], title=None,
                freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}, figsize=(16, 6),
                color_palette="blend:#8e44ad,#3498db", n_colors=2,show=True,save=False, filename="tests\\figures\\erdslines.png", dpi=300)

###********** MAPAS TOPOGRÁFICOS **********###

# Configuración de parámetros para computar la TFR
tmin = -2
tmax = 4
dt = 0.5
fmin = 5     # Frecuencia mínima de interés
fmax = 36    # Frecuencia máxima de interés

##computo la TFR para todos los electrodos
tfr, freqs, times = compute_tfr(
    eeg_data_reconstructed, event_times_samples, event_labels, tmin-dt, tmax+dt, fmin, fmax, n_cycles=20,
    reject=dict(eeg=120), baseline=(-2,-1), baseline_mode="percent", baseline_cropping=(-2,2) )

##seteo el montaje
tfr.info.set_montage(montage)
print(tfr.info['dig'])

##grafico los mapas topográficos
times=[-1.0,-0.5, 0, 0.5,1] ##tiempos en segundos
class_interest = "DERECHA"

plot_ERDS_topomap(
    tfr=tfr,
    times=times,
    class_interest=class_interest,
    bands_interest=["alpha", "beta"],
    freq_bounds={"alpha": (7, 14), "beta": (14, 35)},
    title=f"Mapas Topográficos ERDS - Mano: {class_interest}",
    show=True,
    save=False, filename="tests\\figures\\erds_topomap.png",
    apply_cnorm=True,
    vmin=-1.5,
    vmax=1,
    cmap="PiYG",
    colorbar=True,
    show_names=False,
    figsize=(16, 12),
    )