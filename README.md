# Five Star - The General Classifier

The aim of this project is to be a one stop shop for labelling, cleaning and augmenting data, and then training, testing and using a deep learning classifier.

Everything you need is easily configured in the config file. No need for any code changes!

# Setup
1. This project requires python 3.6 or greater. It is recommended that you use [virtualenv](https://thepythonguru.com/python-virtualenv-guide/) to set up an environment specifically for this project, although this is not required. Create a python3.6 virtual environment with the following command:
```
virtualenv ~/virtualenvs/fivestar -p python3.6
```

2. This tool is built on Pytorch. We recommend preinstalling pytorch using the [pytorch.org installer](https://pytorch.org/get-started/locally/)
if you want pre-install the version of pytorch specific to your CUDA version, or if you want to run it on the CPU. Please run this command before installing the additional dependencies

3. Install the project dependencies:
pip install -r requirements.txt

4. Set up your config.hjson file to point to your dataset and make sure the classes are set. You can also choose whether it should run on the CPU or GPU and which deep learning model it should use. There is an exampleconfig.hjson with default values for all the settings.

# Supported Networks
A list of supported networks can be found [here](networks.md)

# The labelling tool
This is a very simple tool written in Tkinter for quickly labelling images
for classification. Run it on a folder of images and it will move them to
a class folder in the labelled_images folder. It keeps track of which image
you last labelled and it also checks which images were previously labelled.

# Starting from Scratch
1. edit config.hjson to specify the classes, the file ending, the keyboard shortcuts and the image size you want displayed.
2. Use the labelling tool to label your images
    - Specify your classes and keyboard shortcuts in the config file
    - Drop your images into the labelling tool folder
    - run the labelling tool:```python labeller.py```
    - once finished, copy the labelled folders to the training folder specified in the config file
3. Run the splitter in the utils folder to break it into test and train datasets ```python splitter.py```
4. train the network: ```python five-star.py train```
5. run ```tensorboard --logdir runs``` to view the progress of the training
6. The model will be saved and you can now use it to classify images
7. Put images in the unclassified folder (specified in the config file)
8. ```python five-star.py classify```
9. look in the classified folder to see the results

# Starting with an example dataset.
You can download a sample dataset, Microsoft research cats and dogs dataset [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765). If you want to start with a smaller subset of cats and dogs you can download it [here](https://intel-my.sharepoint.com/:u:/p/jonathan_byrne/EasebHZE6LVFmstXQ2fS2tAB2bMkn31scbGNlv6cuz1T0Q?e=oVsyo5)

Extract the images to data/labelled_data and then run the script to split it into train and test folders. The training folder
is used for training and runtime validation. The test folder is *only* used for verification of the model after the run.

Five star expects the data to be in the following format:
<data_folder>/train/<class>
<data_folder>/test/<class>

e.g.;
data/labelled_data/train/cat
data/labelled_data/train/dog
data/labelled_data/test/cat
data/labelled_data/test/dog

# Running five-star
Once your data is all in order all you have to do is:
```
python fivestar.py train
```
Once it is complete you can validate it using:
```
python fivestar.py test
```

If you are happy with the results you can use it to classify images in a pre-specified folder using:
```
python fivestar.py classify
```
set the input (unclassed_data_folder) and output(classed_data_folder) folders in config.json

# Analysing the results:
Five-star outputs tensorboard graphs, you can use tensorboard to view the run in realtime.
```
tensorboard --logdir runs
```
The graphs can be viewed in the browser at: http://localhost:6006/ 

At the end of the run a confusion matrix is generated. If it does not display automatically it should be in the root folder as <network>_train_heatmap.png


# Utilities
## Splitter.py
If the dataset is all in one folder, rather than separate test and train folders, then you must utils/splitter.py to partition the data into train and test folders.**the test data is not involved in the training process at all**, it is used for unbiased evaluation after the training is complete. The partition ratio is specified in the config file as train_test_split.

