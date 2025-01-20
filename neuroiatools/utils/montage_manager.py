import xml.etree.ElementTree as ET

def xml_to_sfp(xml_file, output_file):
    """
    Convierte un archivo XML generado por MontageCreator a un archivo .sfp compatible con MNE.

    Parámetros:
        xml_file (str): Ruta al archivo XML de entrada.
        output_file (str): Ruta al archivo .sfp de salida.
    """
    # Parsear el archivo XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extraer datos de electrodos
    electrode_names = root.find('electrodename').text.split(',')
    x_positions = list(map(float, root.find('xposition').text.split(',')))
    y_positions = list(map(float, root.find('yposition').text.split(',')))
    z_positions = list(map(float, root.find('zposition').text.split(',')))

    # Verificar que las longitudes coincidan
    if not (len(electrode_names) == len(x_positions) == len(y_positions) == len(z_positions)):
        raise ValueError("El número de electrodos no coincide con las posiciones especificadas.")

    # Escribir el archivo .sfp
    with open(output_file, 'w') as sfp_file:
        for name, x, y, z in zip(electrode_names, x_positions, y_positions, z_positions):
            sfp_file.write(f"{name} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Archivo .sfp generado exitosamente en: {output_file}")