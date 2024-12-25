import numpy as np
import logging
import pandas as pd

class TrialsHandler():
    """Clase para obtener los trials a partir de raw data"""

    def __init__(self, rawEEG, timeEvents, sfreq, tmin = -0.2, tmax = 1, reject = None, trialsToRemove = None) -> None:
        """Constructor de la clase Trials
        Parametros:
            - rawEEG (numpy.array): array de numpy con la señal de EEG de la forma [channels, samples]
            - timeEvents (numpy.array): array de numpy con los eventos de tiempo en segundos.
            - sfreq (float): frecuencia de muestreo de la señal de EEG.
            - tmin, tmax: tiempo inicial del trial y tiempo final del trial relativos al timeEvent en segundos.
            - reject (float): Valor de umbral para rechazar trials. Si el valor absoluto de alguno de los canales
            supera este valor, el trial es rechazado. Si es None, no se rechazan trials.
            - trialsToRemove (list): lista de trials a remover. Si es None, no se remueven trials.
            """
        
        if not isinstance(rawEEG, np.ndarray):
            raise TypeError("rawEEG debe ser un array de numpy")
        self.rawEEG = rawEEG
        if not isinstance(timeEvents, np.ndarray):
            raise TypeError("timeEvents debe ser un array de numpy")
        self.timeEvents = timeEvents
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        
        self.trials = self.getTrials() #array de numpy con los trials de la forma [trials, channels, samples]
        #chequeamos si hay trials que remover
        if trialsToRemove is not None:
            self.removeTrials(trialsToRemove)

        self.rejectedTrials = None
        if reject is not None:
            self._rejectTrials()

    def getTrials(self):
        """Función para extraer los trials dentro de self.rawEEG"""

        if self.tmin > self.tmax:
            raise ValueError("tmin debe ser menor que tmax")

        init_idx = np.round((self.timeEvents + self.tmin) * self.sfreq).astype(int) #índices de inicio de cada trial
        end_idx = np.round((self.timeEvents + self.tmax) * self.sfreq).astype(int) #índices de fin de cada trial
        ## concatenamos en time_idx los índices de inicio y fin de cada trial
        time_idx = np.vstack((init_idx, end_idx)).T  # transponemos para que tenga forma (n_trials, 2)

        # Tamaño del segmento esperado
        segment_length = int(round((self.tmax - self.tmin) * self.sfreq)) ##número de muestras en el segmento

        trials = np.zeros((len(init_idx), self.rawEEG.shape[0], segment_length)) ##array de numpy con los trials de la forma [trials, channels, samples]

        for i, (start, end) in enumerate(time_idx):
            if end - start != segment_length:
                raise ValueError(
                    f"El tamaño del segmento ({end - start}) no coincide con el tamaño esperado ({segment_length}) "
                    f"para el trial {i}."
                )
            trials[i] = self.rawEEG[:, start:end]

        return trials
    
    def saveTrials(self, filename):
        """Función para guardar los trials en un archivo .npy"""
        np.save(filename, self.trials)
        print("Se han guardado los trials en {}".format(filename))

    def removeTrials(self, trialsToRemove:list):
        """Función para remover trials usando la lista de trialsToRemove.
        A partir de trialsToRemove removemos los indices de self.timeEvents, luego actualiamos
        self.trials y self.labels"""

        ##chequeamos que trialsToRemove sea una lista
        if not isinstance(trialsToRemove, list):
            raise TypeError("trialsToRemove debe ser una lista")
        
        ##chequeamos que los valores de los trials existan cómo indices
        if not all(trial in self.timeEvents.index for trial in trialsToRemove):
            raise ValueError("Uno o más valores pasados como trials no existen cómo índices en self.timeEvents")
        
        else:
            #removemos los trials de self.timeEvents
            self.timeEvents = self.timeEvents.drop(trialsToRemove)
            #eliminamos los trials de self.trials
            self.trials = np.delete(self.trials, trialsToRemove, axis=0)
            #eliminamos los trials de self.labels
            self.labels = np.delete(self.labels, trialsToRemove, axis=0)

            print("Se han removido los trials {}".format(trialsToRemove))

    def _rejectTrials(self):
        """Función para rechazar trials en base a self.reject.
        Se revisa la amplitud pico a pico de cada trial y si supera el valor de self.reject, se rechaza el trial.

        Revisar https://github.com/mne-tools/mne-python/blob/maint/1.9/mne/epochs.py#L894 para ver cómo se hace en MNE
        
        Retorna una lista con los índices de los trials rechazados"""
        pass
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    ##eeg de prueba
    sfreq = 512
    rawEEG = rawEEG = np.load("datasets\\signal_test_trialshandler.npy")
    timevents = np.load("datasets\\timevents_signal_test_trialshandler.npy")

    ejet = np.arange(0,len(rawEEG[0]))/sfreq
    plt.plot(ejet, rawEEG[0], label="Señal de prueba")
    ##agrego linea vertical en el tiempo del evento
    plt.axvline(x=timevents[0], color='r', linestyle='--', label="Evento")
    plt.title("Señal de prueba")
    plt.legend()
    plt.show()
    
    tmin = -1
    tmax = 3
    trialshandler = TrialsHandler(rawEEG, timevents, sfreq, tmin=tmin, tmax=tmax)

    trials = trialshandler.trials
    ejet = np.arange(0,trials.shape[2])/sfreq+timevents[0]+tmin
    plt.plot(ejet,trials[0,0,:], label="Trial extraído")
    ##agrego una línea vertical en el tiempo del evento
    plt.axvline(x=timevents[0], color='r', linestyle='--', label="Evento")
    ##agrego una línea vertical en tmin
    plt.axvline(x=timevents[0]+tmin, color='g', linestyle='--', label="tmin")
    ##agrego una línea vertical en tmax
    plt.axvline(x=timevents[0]+tmax, color='y', linestyle='--', label="tmax")
    plt.title("Trial extraído")
    plt.legend()
    plt.show()
    