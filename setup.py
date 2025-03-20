from setuptools import setup, find_packages

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='treeIDW', 
    version='1.0.0', 
    url='https://github.com/Dorian210/treeIDW', 
    author='Dorian Bichet', 
    author_email='dbichet@insa-toulouse.fr', 
    description='A package for KD-tree optimized inverse distance weighting interpolation.', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    packages=find_packages(), 
    install_requires=['numpy', 'numba', 'scipy'], 
    classifiers=['Programming Language :: Python :: 3', 
                 'Operating System :: OS Independent', 
                 'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)'], 
    
)
