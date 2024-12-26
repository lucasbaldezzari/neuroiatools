# Neuro-IA Lab tools

<img align="right" src="neuroialogo.png" alt="Neuro-IA Lab" width="210">

Herramientas de procesamiento y análisis de señales de Electroencefalografía (EEG) del *"Laboratorio de Neurociencias e Ingeligencia Artificial Aplicada"* de la [Universidad Tecnológica](https://utec.edu.uy/en/) del Uruguay.

### Paquete

[neuroiatools](https://pypi.org/project/neuroiatools/)

### Autor

- [MSc. BALDEZZARI Lucas](https://www.linkedin.com/in/lucasbaldezzari/) (lucas.baldezzari@utec.edu.uy).

### Última versión

#### 1.1.8

- Se implementa clase Filter para filtrar señales. La clase toma la señal y aplica el filtro en el último eje. Aplica primero un pasabanda y luego un notch. No aplica ventana, de momento no deja elegir entre FIR o IIR, por defecto aplica IIR.
