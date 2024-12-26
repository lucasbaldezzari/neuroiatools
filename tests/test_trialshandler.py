from neuroiatools.EEGManager import TrialsHandler
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##cargo datasets/raweeg_executed_tasks.hdf5
raweeg = h5py.File("datasets\\raweeg_executed_tasks.hdf5", "r")["raw_eeg"][:]
print("raweeg shape: ", raweeg.shape)
eventos = pd.read_csv("datasets\\events_executed_tasks.txt")
print(eventos.head())

sfreq = 512

trial = 1
tarea = eventos.loc[1,"class_name"]
canal = 32

time_events = eventos["event_time"].values/sfreq
trial_time = time_events[trial-1]

tmin = -2
tmax = 4

t_fi = 0.5 #tiempo fade-in
t_cue = 2 #tiempo cue
t_fo = 1 #tiempo fade-out

total_time = t_fi + t_cue + t_fo

##instancio un objeto de la clase TrialsHandler
th = TrialsHandler.TrialsHandler(raweeg, time_events, sfreq, tmin=tmin, tmax=tmax)
trials = th.trials #obtengo los trials

ejet = np.arange(0, trials.shape[2])/sfreq

fig, ax = plt.subplots(figsize=(15, 4))
plt.plot(ejet+trial_time+tmin, trials[trial-1,canal-1,:], color = "black")
##agrego sombras
plt.axvspan(trial_time-t_fi, trial_time, alpha=0.2, color='yellow',label='Fade-in')
plt.axvspan(trial_time, trial_time+t_cue, alpha=0.2, color='gray',label='Cue')
plt.axvspan(trial_time+t_cue, trial_time+t_cue+t_fo, alpha=0.2, color='blue',label='Fade-out')
##linea de cue
plt.axvline(trial_time, color='black', linestyle='--', label='Cue on')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [uV]")
plt.legend(loc="upper right")
plt.title("EEG sin filtrar" + " - Trial "+str(trial)+" - "+tarea + " - Canal "+str(canal))
plt.grid()
plt.show()