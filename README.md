# MLEBE

[![Build Status](https://travis-ci.com/Jimmy2027/MLEBE.svg?branch=master)](https://travis-ci.com/Jimmy2027/MLEBE)

MLEBE (Machine Learning Enabled Brain Extraction) is a python package for both the training of a 3D Unet for segmentation and its integration into the [SAMRI](https://github.com/IBT-FMI/SAMRI) workflow.

## Installation
#### Python Package Manager (Users):
Python's `setuptools` allows you to install Python packages independently of your distribution (or operating system, even).
This approach cannot manage any of our numerous non-Python dependencies (by design) and at the moment will not even manage Python dependencies;
as such, given any other alternative, **we do not recommend this approach**:

````
git clone https://github.com/Jimmy2027/MLEBE.git
cd MLEBE
python setup.py install --user
````

If you are getting a `Permission denied (publickey)` error upon trying to clone, you can either:

* [Add an SSH key](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/) to your GitHub account.
* Pull via the HTTPS link `git clone https://github.com/Jimmy2027/MLEBE.git`.
##### Using a conda venv:
````
git clone https://github.com/Jimmy2027/MLEBE.git
cd MLEBE
path/to/conda/environment/bin/python -m pip install .
````
#### Python Package Manager (Developers):
Python's `setuptools` allows you to install Python packages independently of your distribution (or operating system, even);
it also allows you to install a "live" version of the package - dynamically linking back to the source code.
This permits you to test code (with real module functionality) as you develop it.
This method is sub-par for dependency management (see above notice), but - as a developer - you should be able to manually ensure that your package manager provides the needed packages.

````
git clone git@github.com:Jimmy2027/MLEBE.git
cd MLEBE
echo "export PATH=\$HOME/.local/bin/:\$PATH" >> ~/.bashrc
source ~/.bashrc
python setup.py develop --user
````

If you are getting a `Permission denied (publickey)` error upon trying to clone, you can either:

* [Add an SSH key](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/) to your GitHub account.
* Pull via the HTTPS link `git clone https://github.com/Jimmy2027/MLEBE.git`.

##### Using a conda venv:
````
git clone https://github.com/Jimmy2027/MLEBE.git
cd MLEBE
path/to/conda/environment/bin/python -m pip install -e .
````