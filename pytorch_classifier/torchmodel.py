""" @package five-star
TorchModel - Generic container for pytorch models
for training and inference

@author anton.shmatov@intel.com

@update aoife.harte@intel.com
@date 18/09/20

@update saksham.sinha@intel.com
@date 20/11/2020
"""

import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import warnings
import glob

from torch.utils.data import random_split
from utils.plot_cm import plot_confusion_matrix

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report
from copy import deepcopy
from torchvision import models, transforms, datasets
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from datetime import datetime
from sklearn.metrics import confusion_matrix
import pytorch_classifier.transforms as ctransforms
from PIL import Image

## \brief Constructs a Torch model of a given nn.Module with training facilities
# \details This class accepts any torch-created nn.Module model, along with the input
# dimension as well as number of output classes. The class can then be used
# to train the model or perform inference with it on some input data.
class TorchModel:
    ## Attain some input parameters and create the altered model
    def __init__(self, model=None, name=None, out_chan=4, gpu=True, cpu_cores=None):
        self.name = name
        # testing or classifying
        if isinstance(model, str):
            if gpu:
                print("running on GPU")
                self.model = torch.load(model)
            else:
                if cpu_cores is None:
                    print("Need to specify the number of cores for CPU workload")
                    exit()
                print(f"running on CPU on {cpu_cores} cores")
                self.cpu_cores = cpu_cores
                self.model = torch.load(model, map_location=torch.device("cpu"))
            print("Model loaded successfully")

        # training
        else:
            self.model = model

            self.__expandDense__(out_chan)

            print("New model successfully initialised")

        self.gpu = gpu
        self.cpu_cores = cpu_cores

        # convert model to gpu
        self.cuda(self.model)

        self._separator = "-" * 40

        # load model's class names if they exist already
        self.class_names = getattr(self.model, "class_names", None)

    ## cudarize any variable if needed
    def cuda(self, variable):
        if self.gpu:
            return variable.cuda()
        else:
            return variable.cpu()

    ## Expand the number of output channels for extra classes
    def __expandDense__(self, out_chan):

        # Not the most elegant approach to handling different
        if "efficientnet" in self.name:
            last_feature_name = list(self.model.named_children())[-2][0]
            last_feature = getattr(self.model, last_feature_name)
            sequential = False
            dense_layer = last_feature

            features = [
                nn.BatchNorm1d(dense_layer.in_features),
                nn.Dropout(0.25, inplace=True),
                nn.Linear(dense_layer.in_features, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, out_chan, bias=True),
                nn.Softmax(dim=1),
            ]
        else:
            last_feature_name = list(self.model.named_children())[-1][0]
            last_feature = getattr(self.model, last_feature_name)

            # find out if this feature is a sequential, then extract the dense layer
            if isinstance(last_feature, nn.Sequential):
                sequential = True
                dense_layer = last_feature[-1]
            else:
                sequential = False
                dense_layer = last_feature

            # # check to make sure we found the dense layer
            if not isinstance(dense_layer, nn.Linear):
                raise Exception(f"Could not find last Dense layer for output channel . Found {dense_layer}")

            # assemble the features
            if sequential:
                # in case of sequential, only add the dense and softmax
                features = list(last_feature)[:-1]
                features.append(nn.Linear(dense_layer.in_features, out_chan))
                features.append(nn.Softmax(dim=1))
            else:
                # in case of single layer, add a bit of pizzazz
                features = [
                    nn.BatchNorm1d(dense_layer.in_features),
                    nn.Dropout(0.25, inplace=True),
                    nn.Linear(dense_layer.in_features, 512, bias=True),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, out_chan, bias=True),
                    nn.Softmax(dim=1),
                ]

        # construct the sequential model
        self._classifier_layer = nn.Sequential(*features)

        # set the new dense layer in
        setattr(self.model, last_feature_name, self._classifier_layer)

    ## provide a classification report on the evaluated data
    def __report__(self, predictions, gt):
        ft_report = classification_report(
            gt,
            predictions.argmax(axis=1),
            target_names=self.class_names,
            labels=range(len(self.class_names)),
        )

        return ft_report

    def __loadData__(self, data_transform, val_div=4):
        """
        loads the data from a known folder that conforms to ImageFolder
        """
        if not data_transform:
            data_transform = {
                "train": transforms.Compose(
                    [
                        ctransforms.Resize(self.model.config),
                        transforms.ToTensor(),
                    ]  # AAP resize
                ),
                "val": transforms.Compose(
                    [
                        ctransforms.Resize(self.model.config),
                        transforms.ToTensor(),
                    ]  # AAP resize
                ),
            }
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        image_dataset = datasets.ImageFolder(self._hyperparams["data_folder"], transform=transform)
        total_images = len(image_dataset)
        self._dataloaders = {}

        # check if testing or training
        if self._hyperparams["l_rate"] == 0:
            self._dataloaders["val"] = torch.utils.data.DataLoader(
                image_dataset,
                batch_size=self._hyperparams["batch_size"],
                shuffle=True,
                num_workers=self.cpu_cores,
            )
        else:
            split = self._hyperparams["split"]
            train_images = int(len(image_dataset) * (1 - split))
            val_images = int(len(image_dataset) * split)

            if train_images + val_images < len(image_dataset):
                train_data, val_data = random_split(
                    image_dataset,
                    [train_images, val_images + 1],
                    generator=torch.Generator().manual_seed(42),
                )
            else:
                train_data, val_data = random_split(
                    image_dataset,
                    [train_images, val_images],
                    generator=torch.Generator().manual_seed(42),
                )

            self._dataloaders["train"] = torch.utils.data.DataLoader(
                train_data,
                batch_size=self._hyperparams["batch_size"],
                shuffle=True,
                num_workers=self.cpu_cores,
            )
            self._dataloaders["val"] = torch.utils.data.DataLoader(
                val_data,
                batch_size=self._hyperparams["batch_size"],
                shuffle=True,
                num_workers=self.cpu_cores,
            )

        self.class_names = image_dataset.classes
        print(f"\nClasses: {self.class_names}")

    ## get the weights for a certain set of classes
    def __getClassWeights__(self):
        print(self._hyperparams["data_folder"])
        files_path = os.path.join(
            self._hyperparams["data_folder"],
            "*",
            "*." + self._hyperparams["file_ending"],
        )
        filelist = glob.glob(files_path)
        labels_files_dict = {}
        labels_set = set()
        for file in filelist:
            filename = os.path.basename(file)
            label = os.path.basename(os.path.dirname(file))
            if label not in labels_set:
                labels_set.add(label)
                labels_files_dict[label] = 1
            labels_files_dict[label] += 1
        max_class = max(labels_files_dict.values())
        for l in labels_files_dict:
            labels_files_dict[l] = max_class / labels_files_dict[l]
        print("Class weights:", labels_files_dict)
        distribution = list()

        for name in self.class_names:
            distribution.append(labels_files_dict[name])

        return self.cuda(torch.tensor(distribution, dtype=torch.float))

    ## initialize the optimisers
    def __initOpt__(self, weight=None):
        """
        Initialise the optimizer and the scheduler
        """

        # initialise some training parameters
        if not hasattr(self, "_criterion"):
            if weight is None:
                self._criterion = nn.CrossEntropyLoss()
            else:
                print(f"Using class weights: {weight}")
                self._criterion = nn.CrossEntropyLoss(weight=weight)

            self._optimizer = optim.Adam(self.model.parameters(), lr=self._hyperparams["l_rate"])
            self._scheduler = lr_scheduler.CyclicLR(
                self._optimizer,
                base_lr=self._hyperparams["l_rate"],
                max_lr=0.2,
                step_size_up=100 * self._hyperparams["batch_size"],
                step_size_down=250 * self._hyperparams["batch_size"],
                cycle_momentum=False,
            )

    def __freezeFeatures__(self, freeze=True, features=None):
        """
        Freeze or unfreeze a particular set of features within the model
        by default performs freeze on all features
        """
        if features is None:
            print(f"{'' if freeze else 'Un'}Freezing feature layers\n")
            features = self.model.children()

        grad = not freeze
        for feature in features:
            feature.requires_grad = grad

            # each feature has a set of parameters
            for param in feature.parameters():
                param.requires_grad = grad

    def __freezeClassifier__(self, freeze=True):
        """
        freeze or unfreeze the classifer features
        """
        print(f"{'' if freeze else 'Un'}Freezing classifier layers\n")

        self.__freezeFeatures__(freeze=freeze, features=[self._classifier_layer])

    def __freezeTopFeatures__(self, freeze=True, features=None):
        """
        freeze or unfreeze the top third of a network, by default performed on
        the standard features to affect the conv layers
        """
        print(f"{'' if freeze else 'Un'}Freezing Top layers\n")

        if features is None:
            features = list(self.model.children())

        # calculate the third and freeze those
        amount = len(features) // 3
        self.__freezeFeatures__(freeze=freeze, features=features[-amount:])

    ## begin training the model on the data given in the folder
    def train(
        self,
        data_folder,
        data_transform=None,
        batch_size=32,
        epochs=100,
        train_set="train",
        split=0.2,
        l_rate=0.0001,
        config={},
        name="no_name",
        heatmap_name="heatmap.png",
    ):

        self._hyperparams = {
            "data_folder": data_folder,
            "batch_size": batch_size,
            "epochs": epochs,
            "split": config["train_val_split"],
            "file_ending": config["file_ending"],
            "train_set": train_set,
            "l_rate": l_rate,
            "config": config,
            "name": name,
        }

        print(self.model)

        # print some data about the training
        print(
            f"Beginning training on model with:\nbatch_size:{batch_size}\n\
                \repochs:{epochs}\nlearning_rate:{l_rate}\nTraining set:{train_set}\n"
        )

        # save the config within the model
        setattr(self.model, "config", config)
        setattr(self.model, "_hyperparams", self._hyperparams)

        # load the data
        self.__loadData__(data_transform)

        # freeze features but not classifier
        self.__freezeFeatures__(True)
        self.__freezeTopFeatures__(True)
        self.__freezeClassifier__(False)

        class_weights = self.__getClassWeights__()

        # initialize optimizers
        self.__initOpt__(weight=class_weights)

        # save the class names within the model itself
        setattr(self.model, "class_names", self.class_names)

        # set up tensorboard
        writer = SummaryWriter()

        # perform initial validation
        self.eval(report=True)

        # begin training
        since = time.time()

        # store a copy of the network
        best_model = deepcopy(self.model.state_dict())

        # number of batches
        train_batches = len(self._dataloaders[train_set])

        # training values
        loss_train = 0
        acc_train = 0
        best_acc = 0
        best_loss = 100
        avg_val_loss = 0

        # step once before training begins
        self._optimizer.step()

        # iterate through epochs
        for epoch in range(epochs):
            self.__freezeClassifier__(False)

            # display epoch
            print(f"Epoch {epoch + 1}/{epochs}")
            print(self._separator)

            # set training mode to true and optimize parameters
            self.model.train(True)
            torch.enable_grad()

            # iterate through batches
            for i, data in enumerate(self._dataloaders["train"]):

                # alter learning rate
                # self._scheduler.step()

                # print details of current batch training
                print(
                    f"\rTraining batch {i + 1}/{train_batches}; loss {loss_train:0.4f} acc {acc_train:0.4f}",
                    end="",
                )

                # convert data to cuda variables
                inputs, labels = data
                inputs, labels = Variable(self.cuda(inputs)), Variable(self.cuda(labels))
                # reset gradients
                self._optimizer.zero_grad()

                # forward pass
                outputs = self.model(inputs)

                # calculate loss
                _, preds = torch.max(outputs, 1)
                loss = self._criterion(outputs, labels)

                # perform back prop and optimize
                loss.backward()
                self._optimizer.step()

                # update parameters with moving window
                window_rate = 0.15
                loss_train = (1 - window_rate) * loss_train + window_rate * (loss.item() / batch_size)
                acc_train = (
                    acc_train * (1 - window_rate) + window_rate * torch.sum(preds == labels).item() / batch_size
                )

            # skip to next line
            print()

            # freeze again so that eval doesn't use all the memory
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            torch.cuda.empty_cache()

            # evaluate
            avg_val_loss, avg_val_acc = self.eval(report=(epoch + 1) % 5 == 0)

            current_lr = self._optimizer.state_dict()["param_groups"][0]["lr"]

            if (epoch + 1) % 5 == 0:
                print(f"Current LR: {current_lr}")

            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Acc/val", avg_val_acc, epoch)
            writer.add_scalar("Loss/train", loss_train, epoch)
            writer.add_scalar("Acc/train", acc_train, epoch)
            writer.add_scalar("learning_rate", current_lr, epoch)

            # check if evaluation got better, if so then save new details
            if avg_val_loss <= best_loss and (avg_val_acc - best_acc) >= -1e-4:
                best_acc = avg_val_acc
                best_loss = avg_val_loss
                best_model = deepcopy(self.model.state_dict())

        # print final details of training
        elapsed_time = time.time() - since
        print()
        print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Final results:\nBest loss: {:0.4f}\nBest acc: {:0.4f}\n".format(best_loss, best_acc))

        # load best model in case it wasn't the current one
        self.model.load_state_dict(best_model)

        self.eval(
            report=True, write=True, confmat=True, heatmap_name=heatmap_name
        )  ##aoife - include confusion matrix and heatmap

    # evaluate the performance of the model on the test dataset
    def eval(
        self,
        test_set="val",
        report=False,
        write=False,
        confmat=False,
        heatmap_name="heatmap.png",
        classify=False,
        pred_classes_name="predicted_classes",
    ):
        # keep track of the time and metrics
        since = time.time()
        avg_loss = 0
        avg_acc = 0
        loss_test = 0
        acc_test = 0

        test_batches = len(self._dataloaders[test_set])

        print("Evaluating Model")
        print(self._separator)

        # set model for evaluation
        self.model.train(False)
        self.model.eval()
        torch.no_grad()

        ##save images into predicted classes - aoife
        # make all the relevant folders for output
        if classify:
            if not os.path.isdir(pred_classes_name):
                os.makedirs(pred_classes_name)
            for item in self.class_names:
                classpath = os.path.join(pred_classes_name, item)
                if not os.path.isdir(classpath):
                    os.makedirs(classpath)
            counter = 1

        pred_total = None
        label_total = None
        c_pred_total = None  ##aoife - set actual and prediction variables
        c_label_total = None  ##aoife
        # iterate through batches until we're done the test set
        for i, data in enumerate(self._dataloaders[test_set]):
            print("\rTest batch {}/{}".format(i + 1, test_batches), end="", flush=True)

            # convert input data to variables
            inputs, labels = data

            inputs, labels = Variable(self.cuda(inputs)), Variable(self.cuda(labels))

            # run it through the model
            outputs = self.model(inputs)
            # get the prediction maximum values for each class
            _, preds = torch.max(outputs, 1)

            if pred_total is not None and report:
                pred_total = np.concatenate([pred_total, outputs.detach().cpu().numpy()], axis=0)
                label_total = np.concatenate([label_total, labels.detach().cpu().numpy()], axis=0)
            elif report:
                pred_total = outputs.detach().cpu().numpy()
                label_total = labels.detach().cpu().numpy()
            ##confusion matrix - aoife
            ##create arrays to actual and predicted classes of the images
            if c_pred_total is not None and confmat:
                c_pred_total = np.concatenate([c_pred_total, preds.detach().cpu().numpy()], axis=0)
                c_label_total = np.concatenate([c_label_total, labels.detach().cpu().numpy()], axis=0)
            elif confmat:
                c_pred_total = preds.detach().cpu().numpy()
                c_label_total = labels.detach().cpu().numpy()

            ##save predicted classes - aoife
            if classify:
                ##initialise variables
                img = None
                pixels = None
                i = 0
                ##for every input image
                while i < len(labels):
                    ##load the image by creating new image and setting the individual pixels to the correct colours
                    img = Image.new(mode="RGB", size=(300, 300))
                    pixels = img.load()
                    y = 0
                    while y < 300:
                        x = 0
                        while x < 300:
                            pixels[x, y] = (
                                (inputs[i][2][y][x]) * 255,
                                (inputs[i][1][y][x]) * 255,
                                (inputs[i][0][y][x]) * 255,
                            )
                            x += 1
                        y += 1
                    ##save the image
                    img.save(
                        "%s/%s/%s_predicted_%s_%s.png"
                        % (
                            pred_classes_name,
                            self.class_names[preds[i]],
                            self.class_names[labels[i]],
                            self.class_names[preds[i]],
                            counter,
                        )
                    )
                    ##reset the image
                    img = None
                    pixels = None
                    ##increment the counters
                    i += 1
                    counter += 1

            # calculate loss
            loss = self._criterion(outputs, labels)

            # keep track of metrics
            loss_test += loss.item()
            acc_test += torch.sum(preds == labels).item()

            # del inputs, labels, outputs, preds, data, _
            # if self.gpu:
            #     torch.cuda.empty_cache()

        # get final results
        avg_loss = loss_test / len(self._dataloaders[test_set].dataset)
        avg_acc = acc_test / len(self._dataloaders[test_set].dataset)

        elapsed_time = time.time() - since

        print()
        print(
            "Evaluation completed in {:0.0f}m {:0.0f}s; Avg loss {:.4f}; Avg acc {:.4f}".format(
                elapsed_time // 60, elapsed_time % 60, avg_loss, avg_acc
            )
        )
        print(self._separator)

        if write or report:
            report_details = self.__report__(np.array(pred_total), np.array(label_total))

        if report:
            print(report_details)

        if write:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname = "runs/{}_{}.txt".format(self._hyperparams["name"], now)
            print("saving results to:", fname)
            with open(fname, "w") as f:

                f.write(str(self._hyperparams))
                f.write(report_details)

        ##confusion matrix - aoife
        ##create the confusion matrix
        if confmat:
            ##set actual and predicted variables
            y_true = np.array(c_label_total)
            y_pred = np.array(c_pred_total)
            ##build and print confusion matrix
            conf_mat = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:")
            print(conf_mat)

            # code changed by Saksham
            categories = self.class_names
            plot_confusion_matrix(conf_mat, categories=categories, path_to_save=heatmap_name, cmap="Blues")

        return avg_loss, avg_acc

    ## use the current model to infer on a set of inputs
    def predict(self, inputs, data_format="NHWC"):
        # if only one image then turn into list
        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, 2)

        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, 0)

        # convert data format to NCHW
        if data_format == "NHWC":
            inputs = np.moveaxis(inputs, [0, 1, 2, 3], [0, 2, 3, 1])

        if inputs.dtype == np.uint8:
            inputs = inputs.astype("float64") / 255

        # ensure model is in eval mode
        self.model.eval()
        torch.no_grad()

        # convert to variables
        batch = self.cuda(Variable(torch.tensor(inputs, dtype=torch.float)))

        ##generate prediction from output
        output = self.model(batch)
        _, pred = torch.max(output, 1)
        ##return prediction, not output
        return pred.detach().cpu().numpy()

    def test(
        self,
        data_folder,
        data_transform=None,
        config={},
        batch_size=32,
        save_classes=False,
        heatmap_name="heatmap.png",
        pred_classes_name="predicted_classes",
    ):
        """
        Perform inference on the test set, showing results
        """
        # calculate the class weights
        class_weights = None

        self._hyperparams = {
            "data_folder": data_folder,
            "batch_size": batch_size,
            "config": config,
            "l_rate": 0,
        }

        # initialize optimizers
        self.__initOpt__(weight=class_weights)
        self.__loadData__(data_transform)
        self.eval(
            test_set="val",
            report=True,
            confmat=True,
            classify=save_classes,
            heatmap_name=heatmap_name,
            pred_classes_name=pred_classes_name,
        )

    # return the prediction together with the class names in a dictionary format
    def predict_class(self, *args, **kwargs):
        prediction = self.predict(*args, **kwargs)

        prediction_dict = {class_name: prediction[:, idx] for idx, class_name in enumerate(self.class_names)}

        prediction_dict["best_class"] = [self.class_names[best] for best in prediction.argmax(axis=1)]

        return prediction_dict
