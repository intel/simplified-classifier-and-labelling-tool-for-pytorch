# Five Star: The General Classifier to rule them all
Running on CPU or GPU, windows or linux, pytorch or keras? Five star doesn't care!
Just edit the config file and you're good to go!

should we use a setup.py?

The Intel defect detector code enables the training and execution of deep learning models on image data with minimal use input.

# Set up
It is recommended that you use [virtualenv](https://thepythonguru.com/python-virtualenv-guide/) to set up
an environment specifically for this project.

Optionally you pre-install the version of torch specific to your CUDA version and machine. There is a tool for creating the install command at [pytorch.org](https://pytorch.org)

Once you have done that install the project dependencies:

```pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html```

# Quick start
If you want to classify a folder of images download a trained model [here](https://intel-my.sharepoint.com/:u:/p/jonathan_byrne/EV61H4Jp9g9Em0Qtkw_hMF8B6UniN0CCX9XvvGpiNF1PrQ?e=vk7dEJ) and put it in the models folder. Edit the config.hjson to enable or disable the gpu. Then run

```python use_classifier.py  <folder_name>```

# Overview

- train_classifier.py: Train a resnet50 model based on the provided classes

- test_classifier.py: Evaluate the performance of the trained model on an unseen dataset and generate a confusion matrix of the results

- use_classifier.py: Label a folder of unseen images and output it into the relevant class folders in the results directory.

- config.hjson: contains parameters for the machine, training, and output folders

There are also some helper classes for labelling and tidying the data

# Dataset preparation
This holds the files required to prepare a dataset. Visit this folder if looking to build the dataset before training. It includes files for: image labelling, text removal, data augmentation and duplicate removal.

# labelling tool
this is a simple tkinter gui that can be used for labelling data.

for more information on the code please read the [training guide](training.md)
