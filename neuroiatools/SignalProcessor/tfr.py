import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import pandas as pd

def compute_tfr(data, sfreq, event_times, event_labels, tmin, tmax, fmin, fmax, n_cycles=7, decim=1, num_freqs=30,
                    channel_names=None, pick_channels=None, reject=None, baseline=None, baseline_mode = "percent",
                    baseline_cropping = (-1.5,0.5)):
    """
    Esta función calcula la representación de frecuencia de tiempo (TFR) de datos EEG epocados.
    La TFR es una técnica utilizada para analizar cómo cambia la actividad cerebral a lo largo
    del tiempo en diferentes frecuencias.

    Se hace uso de la librería MNE dado que tiene optimizado el cálculo de la TFR.

    Parameters
    ----------
    data : np.ndarray
        Datos EEG de entrada con forma (n_canales, n_muestras).
    sfreq : float
        Frecuencia de muestreo de los datos en Hz.
    event_times : np.ndarray
        Array con los índices de muestra de los eventos.
    event_labels : list o array
        Nombres de los eventos (str)
    tmin : float
        Tiempo de inicio antes de cada evento en segundos.
    tmax : float
        Tiempo final después de cada evento en segundos.
    dt : float
        Delta t
    fmin : float
        Frecuencia mínima de interés para la TFR en Hz.
    fmax : float
        Frecuencia máxima de interés para la TFR en Hz.
    n_cycles : int, optional
        Número de ciclos por frecuencia para la wavelet de Morlet (opcional, por defecto 7).
    decim : int, optional
        Factor de diezmado para reducir el tamaño de la TFR (opcional, por defecto 1).
    num_freqs: int
        Frecuencias entre fmin y fmax.
    channel_names: list
        Lista con los nombres de los canales a cargar cuando se generen las épocas. Por defecto es None. Cuando es None se generan nombres.
    pick_channels: tuple
        Tupla con los canales a seleccionar (str). Por defecto es None
    reject: dict|None
        Diccionario con el key "eeg" y el value con el valor de voltaje a tomar como pico a pico para rechazar las épocas.
        Por defecto es None.
    baseline: tuple|None
        Tupla con los tiempos de inicio y fin del baseline. Por defecto es None.
    baseline_mode: str
        Modo de la línea base. Por defecto es "percent".
    baseline_cropping: tuple
        Tupla con los tiempos de inicio y fin para aplicar el cropping a cada época antes de aplicar el baseline. Por defecto es (-1.5,0.5).

    Returns
    -------
    tfr : mne.time_frequency.AverageTFR
        La representación de frecuencia de tiempo calculada.
    freqs : np.ndarray
        Frecuencias utilizadas para la TFR en Hz.
    times : np.ndarray
        Puntos de tiempo utilizados para la TFR en segundos.
    """
    ##validamos dimensión
    if data.ndim != 2:
        raise ValueError("Los datos de entrada deben tener la forma (n_channels, n_samples).")

    ## Creamos un objeto mne
    n_channels, n_samples = data.shape
    if channel_names is None:
        channel_names = [f"{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")

    ## objeto mne.raw
    raw = mne.io.RawArray(data, info)

        ## Map event labels to integer IDs
    unique_labels = np.unique(event_labels)
    event_id = {label: idx for idx, label in enumerate(unique_labels)}
    print("Labels de cada evento:",event_id)

    ## array con los eventos. Debe tener la forma (n_events, 3), ver documentación en https://mne.tools/stable/generated/mne.Epochs.html#mne-epochs
    events = np.array([[int(time), 0, event_id[label]] for time, label in zip(event_times, event_labels)])

    ## objeto mneEpochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, detrend=1, preload=False, picks=pick_channels,reject=reject)

    #definimos las frecuencias de interes
    freqs = np.linspace(fmin, fmax, num=num_freqs) ##cantidad de frecuencias entre fmin y fmax

    # Computamos usando Morlet

    tfr = epochs.compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=decim,
    )

    if baseline is not None:
        # tfr.apply_baseline(baseline, mode=baseline_mode, baseline_times=baseline_times)
        tmin_crop, tmax_crop = baseline_cropping
        tfr.crop(tmin_crop, tmax_crop).apply_baseline(baseline, mode=baseline_mode)

    ## Retornamos la TRF, las frecuencas y los puntos temporales
    return tfr, tfr.freqs, tfr.times


def plotTFRERDS(tfr,event_ids=dict(IZQUIERDA=1, DERECHA=2), ch_names = ("C3","Cz","C4"), vmin=-1, vmax=1.5, titles=None,
                show=True, save=False, files_names=None, dpi=300, figsize=(12, 4)):
    """
    Función para crear gráficos tiempo/frecuencia a partir del cálculo la TRF. 
    Esta función está basado en el ejemplo: https://mne.tools/1.7/auto_examples/time_frequency/time_frequency_erds.html

    Parametros:
    ----------
    tfr : Instancia de EpochsTFR o AverageTFR
        La función neuroiatools.SignalProcessor.tfr retorna el objeto correspondiente.
    event_ids : dict
        Diccionario con los nombres de los eventos y sus respectivos índices.
    ch_names : list

    vmin : float
        Valor mínimo para la escala de color.
    vmax : float
        Valor máximo para la escala de color.
    title : list|None
        Título de la figura. Debe ser una lista con los títulos de cada subplot. Deben haber tantos nombres como eventos.
        Si es None se genera un título por defecto.
    show : bool
        Indica si se debe mostrar la figura.
    save : bool
        Indica si se debe guardar la figura.
    files_names : list|None
        Nombre de los archivos de salida. Debe ser una lista con los nombres de cada archivo. Deben haber tantos nombres como eventos.
        Si es None se genera un nombre para cada gráfica.
    dpi : int
        Resolución de la imagen en puntos por pulgada.
    figsize : tuple
        Tamaño de la figura en pulgadas.
    """
    
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # mínimo, centro y máximo para los ERDS
    kwargs = dict(
        n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
    )

    for i, event in enumerate(event_ids):
        #Seleccionamos las épocas para la graficación
        tfr_ev = tfr[event]
        gridspec_kw = {"width_ratios": [10 if i < len(ch_names) else 1 for i in range(len(ch_names)+1)]}
        fig, axes = plt.subplots(
            1, len(ch_names)+1, figsize=figsize, gridspec_kw=gridspec_kw)
        
        for ch, ax in enumerate(axes[:-1]):
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

            c = np.stack(c1 + c2, axis=2)
            p = np.concatenate((p1, p2))
            mask = c[..., p <= 0.05].any(axis=-1)

            ## Graficamos los TRF
            tfr_ev.average().plot(
                [ch],
                cmap="RdBu",
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask,
                mask_style="mask",
            )

            ax.set_title(ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        
        # fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        # Crear barra de colores con etiquetas personalizadas
        cbar = fig.colorbar(axes[0].images[-1], cax=axes[-1])
        cbar.ax.set_yscale("linear")  # Escala lineal en la barra de colores
        cbar.ax.set_title("ERS", fontsize=10, pad=10)  # Texto encima de la barra
        cbar.ax.set_xlabel("ERD", fontsize=10, labelpad=10)  # Texto debajo de la barra

        #títulos de gráficas
        if titles is not None:
            fig.suptitle(titles[i], fontsize=16)
        else:
            fig.suptitle(f"Tiempo/Frecuencia para ({event})", fontsize=16)

        if save:
            if files_names is None:
                filename = f"tfr_{event}_ch{ch_names[i]}.png"
                fig.savefig(filename, dpi=300)
            else:
                print("Guardando figura en:",files_names[i])
                fig.savefig(files_names[i], dpi=dpi)
        if show:
            plt.show()

def plotERDSLines(tfr, channels_order, bands_interest=["alpha", "beta"], title=None,
                freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140},
                color_palette="blend:#8e44ad,#3498db", n_colors=2,show=True,save=False, filename=None, dpi=300, figsize=(12, 4)):
    """Función para graficar las curvas ERDS en un gráfico de líneas.
    El código está basado en el ejemplo:https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html
    
    Parámetros:
    -----------
    tfr : mne.time_frequency.AverageTFR
        La representación de frecuencia de tiempo calculada.
    channels_order : list
        Lista con los nombres de los canales en el orden deseado.
    bands_interest : list
        Lista con las bandas de frecuencia de interés.
    title : str|None
        Título de la figura. Si es None se genera un título por defecto.
    freq_bounds : dict
        Diccionario con los límites de las bandas de frecuencia.
    color_palette : str
        Paleta de colores para las líneas.
    n_colors : int
        Número de colores en la paleta.
    show : bool
        Indica si se debe mostrar la figura.
    save : bool
        Indica si se debe guardar la figura.
    filename : str|None
        Nombre del archivo de salida. Si es None se genera un nombre por defecto.
    dpi : int
        Resolución de la imagen en puntos por pulgada.
    figsize : tuple
        Tamaño de la figura en pulgadas.
        """

    df = tfr.to_data_frame(time_format=None, long_format=True) ##convertimos a dataframe
    print(df.head())

    df["band"] = pd.cut(
        df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]) ##categorizamos las bandas de frecuencia

    ## Filtro para mantener solo las bandas de frecuencia relevantes
    df = df[df.band.isin(bands_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    ## Reordenamos los canales
    df["channel"] = df["channel"].cat.reorder_categories(channels_order, ordered=True)

    plt.figure(figsize=figsize) ##tamaño de la figura
    palette = sns.color_palette(color_palette,  n_colors = n_colors) ##paleta de colores
    g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True, height=4, aspect=2) ##grilla de gráficos
    g.map(sns.lineplot, "time", "value", "condition", n_boot=10,palette=palette) ##graficamos las líneas
    axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5) ##configuración de las líneas
    g.map(plt.axhline, y=0, **axline_kw) ##línea horizontal
    g.map(plt.axvline, x=0, **axline_kw) ##línea vertical
    g.set_axis_labels("Tiempo (s)", "ERDS") ##etiquetas de los ejes
    g.set_titles(col_template="{col_name}", row_template="{row_name}") ##títulos de las gráficas
    g.add_legend(ncol=2, loc="lower center", fontsize=12) ##agregamos la leyenda
    
    ##título de la figura
    if title is None:
        title = "Curvas ERDS%"
    g.figure.suptitle(title, fontsize=16)

    ##guardar la figura
    if save:
        if filename is None:
            filename = "ERDS_lines.png"
        g.savefig(filename, dpi=dpi)
    g.figure.subplots_adjust(top=0.9)

    ##mostrar la figura
    if show:
        plt.show()

def plot_ERDS_topomap(tfr, times, class_interest = None, bands_interest=["alpha", "beta"],
                      freq_bounds=None, apply_cnorm=False, vmin=-1, vmax=1.5, title=None, show=True, save=False, filename=None, figsize=(10,10), dpi=300, **kwargs):
    """
    Función para generar mapas topográficos y así visualizar patrones ERD/ERS a partir de datos de tiempo-frecuencia.

    Parámetros
    ----------
    tfr : mne.time_frequency.EpochsTFR
        La representación de tiempo-frecuencia de los datos.
    times : lista o np.ndarray
        Lista de puntos de tiempo (en segundos) para trazar los mapas topográficos.
    class_interest : str, opcional
        Clase de interés. Por defecto es None.
    bands_interest : lista
        Lista de bandas de frecuencia de interés.
    freq_bounds : dict, opcional
        Diccionario con límites de banda de frecuencia, por ejemplo, {"delta": (0, 3), "theta": (4, 7), "alpha": (8, 13)}. 
        Por defecto es None.
    apply_cnorm : bool
        Indica si se debe aplicar una normalización personalizada a la escala de color.
    vmin : float
        Valor mínimo para la escala de color.
    vmax : float
        Valor máximo para la escala de color.
    title : str, opcional
        Título de la figura. Por defecto es None.
    show : bool
        Indica si se debe mostrar la figura.
    save : bool
        Indica si se debe guardar la figura.
    filename : str, opcional
        Nombre de archivo para guardar la figura. Por defecto es None.
    dpi : int
        Resolución de la imagen guardada en puntos por pulgada.
    kwargs : dict
        Argumentos clave adicionales para la gráfica.

    Retorna
    -------
    None
    """
    if freq_bounds is None:
        freq_bounds = {"delta": (0, 3), "theta": (4, 7), "alpha": (8, 13), "beta": (14, 30), "gamma": (31, 100)}
    
    ## Selección de la clase de interés
    if class_interest is not None:
        tfr = tfr[class_interest]

    ## Verificar si la TFR contiene múltiples épocas
    if tfr.data.ndim == 4:
        ## Promediamos las épocas
        tfr = tfr.average()

    ## Normalización de colores
    if apply_cnorm:
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # mínimo, centro y máximo para los ERDS
        vlim = (None, None)
    else:
        cnorm = None
        vlim = (vmin, vmax)

    ## Creamos la figura
    fig, axes = plt.subplots(len(bands_interest), len(times), figsize=figsize)

    if len(bands_interest) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(times) == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, band in enumerate(bands_interest):
        fmin, fmax = freq_bounds.get(band, (None, None))
        if fmin is None or fmax is None:
            raise ValueError(f"La banda de frecuencia {band} no se encuentra en freq_bounds.")

        for j, time in enumerate(times):
            ax = axes[i, j]
            tfr.plot_topomap(
                tmin=time, tmax=time, fmin=fmin, fmax=fmax, axes=ax, show=False, cnorm=cnorm, vlim=vlim, **kwargs)
            ax.set_title(f"{band.capitalize()} banda\n{time:.2f}s", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save and filename:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()

if __name__ == "__main__":
    from neuroiatools.SignalProcessor import Filter
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
                show=True, save=True, files_names=files_names, dpi=300, figsize=(16, 6))
    
    plotERDSLines(tfr, channels_order=channels, bands_interest=["alpha", "beta"], title=None,
                    freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}, figsize=(16, 6),
                    color_palette="blend:#8e44ad,#3498db", n_colors=2,show=False,save=True, filename="tests\\erdslines.png", dpi=300)