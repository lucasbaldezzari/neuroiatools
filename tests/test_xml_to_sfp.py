from neuroiatools.utils.montage_manager import xml_to_sfp
import mne

xml_to_sfp("tests\\default_montage.xml", "tests\\montage.sfp")

##cargo el archivo .sfp 
montage = mne.channels.read_custom_montage("tests\\montage.sfp")
montage.ch_names