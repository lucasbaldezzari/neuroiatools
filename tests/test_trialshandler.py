from neuroiatools.EEGManager import TrialsHandler
import h5py
import pandas as pd
import matplotlib.pyplot as plt

##cargo datasets/raweeg_executed_tasks.hdf5
raweeg = h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")["raw_eeg"][:]
print("raweeg shape: ", raweeg.shape)
eventos = pd.read_csv("datasets\\events_executed_tasks.txt")
print(eventos.head())

sfreq = 512

time_events = eventos["event_time"].values/sfreq

##instancio un objeto de la clase TrialsHandler
th = TrialsHandler.TrialsHandler(raweeg, time_events, sfreq, tmin=-1, tmax=4)
trials = th.trials

plt.plot(trials[0,0,:])
plt.show()