 # gentriNet
Tutorials and sources to reproduce results obtained during my 2017 summer internship at uOttawa
(WIP)

## Installation

__NB: if you are at uOttawa Department of Geography, all the required libraries are installed in the python evironment located at D:\Amaury\Anaconda3\\__ 

You will need a Python 3 environment with all standard libraries installed. I personally used Anaconda3, and it was my only Python envronment on the machine. CAUTION: When using pip, make sure it is using the pip.exe of the correct Python environment.

#### Determining where your pip.exe is for Anaconda
Open a command prompt in windows by typing "cmd" in the search bar.  In the command prompt type:

```shell 
where conda
``` 

which on my computer returns ```C:\ProgramData\Anaconda3\Scripts\conda.exe```.  As such, in the open command prompt I would type 

```shell
cd C:\ProgramData\Anaconda3\Scripts
``` 
and then 

```shell
dir
```
which returns within the resulting list of files and folders you would see a probram called ```pip.exe```.  So in this folder, if you wanted to install a package for python called 

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
pip install tensorflow_gpu‑1.1.0‑cp36‑cp36m‑win_amd64.whl
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


## Usage example: 
## Main results: Convolutional Neural Networks (CNN)

__Most of the model training and classification was done on a server using a NVIDIA Tesla K80 GPUs. I advise you to use powerful GPUs with more than 10 GB of memory, or else the processings could take a very, very long time__

__If you are at the Geography Department of uOttawa, the set of images is already present on the server in the D:\Amaury\Desktop folder, under the ottawa\_image\_db directory, so do not bother spending several days downloading the StreetView imagery of Ottawa! (yes, this Desktop folder is a bit messy...)__

__Scripts to use: gentriNetConvServer2.py, gentriMap\_allImages.py, to\_positives.py__

Get a .txt file filled with lines like this:
```
path/to/img1.jpg path/to/img2.jpg 0
path/to/img2.jpg path/to/img3.jpg 1
```
with 0 meaning no gentrification between the first and second image and 1 meaning the opposite

Rename this file as retrain.txt and place it in the same directory as __gentriNetConvServer2.py__
You should edit the gentriNetConvServer2.py lines corresponding to learning rates, number of iterations etc. according to your needs.
__Make sure that the last lines are uncommented so the model is saved after you train it for several days...__
### In a more general way, you should read the script files and commentary to understand the way they work before using them

To run the file on the Geography Depatment server, open a command prompt, and type this series of commands:

```shell
D:

cd D:\Amaury\Desktop\

D:\Amaury\Anaconda3\python.exe

[Python shell launches]

>>> exec(open("./gentriNetConvServer2.py").read())
```

Processing can take up to several days, even on a Tesla K80.

Once the model is saved, you can import it to classify your large set of images using __gentriMap\_allImages.py__. You first have to edit the lines to make sure the right model is imported. Use this occasion to read the whole script and make sure it does what you think it does. To run it, if you are in the same command prompt window just type exec(open("./gentriMap\_allImages.py").read())

__CAUTION: you might have a Out Of Memory error (or similar one) if you do too much operations without closing the command prompt window. Workarounds are: close the current cmd window and open another one OR in the "with tf.device('/gpu:x'):" line of the script you want to run, modify the gpu index (0, 1, 2 or 3 instead of x)__

Processing can take up to 30 hours, even on a Tesla K80.

The output of the classification is a sat of 64 files corresponding to the classification results. In order to extract only the positive ones (which are the ones we're interested in) and fuse them in a single file, use the __to\_positives.py__ script. The functions you should use are __res2pos\_multi()__ and __multipos2singlepos()__. Unfortunately, I hardcoded the paths I used on my computer so if you're using another one, maybe you'll have to change them.

The resulting file ("positives\_0-63.txt" by default) is a concatenation of all the positives detected by the model. You can load it in any GIS (I used QGIS) with the fist field as Y coordinates and second field as X coordinates in WGS84.
