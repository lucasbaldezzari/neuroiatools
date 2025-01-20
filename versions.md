#### 1.4.0

- Se implementan los métodos *tfr.plot_ERDS_topomap* en SignalProcessor y *montage_manager.xml_to_sfp* en *utils*. Se crean test para estos métodos.

#### 1.3.0

- Se implementan los métodos *tfr.compute_tfr*, *tfr.plotTFRERDS* y *tfr.plotERDSLines* en SignalProcessor.

#### 1.2.0

- Se implementa método plotEEG.

#### 1.1.9

- Se implementa y testea método _rejectTrials de la clase TrialsHandler. 

#### 1.1.8

- Se implementa clase Filter para filtrar señales. La clase toma la señal y aplica el filtro en el último eje. Aplica primero un pasabanda y luego un notch. No aplica ventana, de momento no deja elegir entre FIR o IIR, por defecto aplica IIR.

#### 1.1.7

- Se modifica TrialsHandler para evitar errores a la hora de obtener los índices iniciales y finales para generar los trials a partir de tmin y tmax.

#### 1.1.6

- Modificaciones menores en *download_dataset* para descargar correctamente los archivos y no en HTML. Se agrega barra de porcentaje descargado.

#### 1.1.5

- Modificaciones menores en *download_dataset* para descargar directamente desde la datasets que se encuentra en la carpeta principal del repositorio.

#### 1.1.4

- Se cambia nombre el nombre del módulo download_dataset por datasetManager

#### 1.1.3

- Se implementa función *download_data* para descargar datos directamente del repositorio. Se elimina función *load_file*.

#### 1.1.2

- Se agrega __init__.py dentro de datasets. Se modifica archivo .toml

#### Versión 1.1.1

- Se modifican nombres de los archivos dentro de datasets.

#### Versión 1.1.0

- Se agrega función *load_file* dentro de un nuevo módulo llamado *utils*. Permite cargar los archivos dentro de la carpeta *datasets*.

#### Versión 1.0.1

- Se corrige readme.md
- Se crea scope en pypi

#### Versión 1.0.0

- Se implementa TrialsHandler.
