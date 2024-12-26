from neuroiatools.utils.datasetManager import download_data
import os
##obtengo la carpeta actual
current_dir = os.getcwd()
download_data(file_name = "events_executed_tasks.txt", save_dir=current_dir+"\\datasets")