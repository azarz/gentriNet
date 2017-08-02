# gentriNet
Tutorials and sources to reproduce results obtained during my 2017 summer internship at uOttawa
(WIP)

## Installation

You will need a Python 3 environment with all standard libraries installed. I personally used Anaconda3, and it was my only Python envronment on the machine. CAUTION: When using pip, make sure it is using the pip.exe of the correct Python environment.

### GIST vectors

To process GIST vectors the way I did in the Python scripts you will find in this repository, you will need to install this implementation: https://github.com/azarz/lear-gist-python. The installation procedure is described in the README.md.

### Google StreetView

The Google API usually doesn't need any installation, but since we want historical imagery, we use this module: https://github.com/robolyst/streetview. To use it, simply place the folder containing the \_\_init\_\_.py in the same directory as your script, and import it using the line 
```python
import streetview
```

### Keras and tensorflow

To install most modules, is use pip.exe, be it through PyPi or through downloaded files. It is the case with Keras, the high-level deep-learning library I used to manipulate neural networks. https://github.com/fchollet/keras

I also used the tensorflow GPU backend. I recommand you start with installing this library.

#### Tensorflow GPU for Windows (Only compatible with NVIDIA GPUs with compute cabability of 3.0 or higher (https://developer.nvidia.com/cuda-gpus))

I found that the steps described in https://www.tensorflow.org/install/install\_windows to not be sufficent for the library to work, so I reccomend you follow these instead:

* Install CUDA® Toolkit 8.0 after downloading the version corresponding to your OS here: https://developer.nvidia.com/cuda-downloads
  * When installing the Toolkit, include the installation of the associated drivers
  * Also make sure that the /bin directory installed along with CUDA is added to the PATH environment variable
* Download cuDNN v5.1 AND cuDNN v6.0 (https://developer.nvidia.com/rdp/cudnn-download) and copy all files from both downloads in the /bin folder of CUDA
* Download tensorflow\_gpu‑1.1.0‑cp36‑cp36m‑win\_amd64.whl from this page: http://www.lfd.uci.edu/~gohlke/pythonlibs/ (this page is extremely useful for Windows users codingin Python)
* Run a command prompt to use pip.exe on that .whl file
```shell
pip install tensorflow\_gpu‑1.1.0‑cp36‑cp36m‑win\_amd64.whl
```

Tensorflow should now be installed on your computer.

#### Keras

Now that tensorflow is installed, you can install Keras uning pip and PyPi
```shell
pip install keras
```

During the installation, you may have prompts that tell you Keras can't be installed because of missing libraries. If this happens, simply use pip to install these missing libraries.
```shell
pip install <missing library name>
```

IMPORTANT: In order to load and save model wieghts (see below), you'll also need to install h5py.
```shell
pip install h5py
```


## Usage

Work in progress
