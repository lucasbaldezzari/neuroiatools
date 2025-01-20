from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor import Filter
from neuroiatools.EEGManager.RawArray import makeRawData
import h5py
import pandas as pd

raw_data = h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")["raw_eeg"][:]
print("raweeg shape: ", raw_data.shape)
eventos = pd.read_csv("datasets\\events_executed_tasks.txt")
print(eventos.head())

sfreq = 512

event_times = eventos["event_time"].values/sfreq ##marcado de eventos en segundos
event_labels = eventos["class_name"].values

raw_eeg = makeRawData(raw_data, sfreq, event_times, event_labels)
##acortamos la señal
raw_eeg.crop(tmin=33)

##grafico el EEG sin filtrar desde el segundo 45
plotEEG(raw_eeg, scalings = 40,show=True, block=True,
        duration = 30, start = 45, remove_dc = True, bad_color = "red")

##filtramos el EEG y luego graficamos
filtro = Filter.Filter(lowcut=0.5, highcut=38, notch_freq=50.0, notch_width=2.0, sample_rate=512.0)
filtered_eeg = filtro.filter_data(raw_data)
filtered_eeg = makeRawData(filtered_eeg, sfreq, event_times, event_labels)
##acortamos la señal
filtered_eeg.crop(tmin=33)

##grafico el EEG sin filtrar desde el segundo 45
plotEEG(filtered_eeg, scalings = 40,show=True, block=True,
        duration = 30, start = 45, remove_dc = True, bad_color = "red")