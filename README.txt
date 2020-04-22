The directory consists of the following subdirectories:

config
---------------
Contains json files with configurations used for the different scripts (e.g. data paths, number of training iterations, max vocabulary size etc.)

esim
---------------
contains utility python scripts - layers.py and model.py define the different pytorch models and their layers, while data.py has some preprocessing functions and utils.py has utility functions.

scripts
---------------
The main python scripts used to load and preprocess the data (under preprocessing), train the models (under training) and evaluate trained models (under testing).