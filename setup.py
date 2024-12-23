from setuptools import setup, find_packages
import toml
import os

def getRequirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    return requirements

def readme():
    with open('README.md') as f:
        return f.read()
    
def version():
    with open('pyproject.toml') as f:
        return toml.load(f)['project']['version']

# Encuentra automáticamente los archivos en 'neuroiatools/datasets'
def find_package_data():
    package_data = {}
    base_path = os.path.join("neuroiatools", "datasets" )
    for root, _, files in os.walk(base_path):
        rel_root = os.path.relpath(root, "neuroiatools")
        package_data[rel_root.replace(os.path.sep, ".")] = files
    return package_data

setup(
    name='neuroiatools',
    version=version(),
    packages=find_packages(),
    install_requires=getRequirements(),
    include_package_data=True,  # Habilita la inclusión de datos no Python
    package_data=find_package_data(),  # Agrega los archivos dentro de datasets
    author='LUCAS BALDEZZARI',
    author_email='lmbaldezzari@gmail.com',
    description='Tools for EEG processing and analysis',
    long_description=readme(),
    long_description_content_type='text/markdown',  # Para renderizar markdown en PyPI
    url='https://github.com/lucasbaldezzari/neuroiatools',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Microsoft :: Windows :: Windows 11',
        'Operating System :: Unix',
    ],
)