import os
import hjson
import torch
import cv2
from tqdm import tqdm
from torchvision import models, transforms
from pytorch_classifier import TorchModel
from pytorch_classifier.transforms import Resize
from efficientnet_pytorch import EfficientNet


def test(config):
    # load the model
    model = TorchModel(
        model=(config["model_path"] + ".trch"),
        out_chan=len(config["classes"]),
        gpu=config["gpu"],
        cpu_cores=config["cpu_cores"],
    )

    test_folder = os.path.join(config["data_folder"], "test")

    print("Testing classifier on images")
    heat_name = config["network"] + "_test_heatmap.png"
    model.test(
        data_folder=test_folder,
        batch_size=config["batch_size"],
        save_classes=config["save_classes"],
        config=config,
        heatmap_name=heat_name,
        pred_classes_name="predicted_classes",
    )


def get_network(network_name, num_classes):
    if "efficient" in network_name:
        network = EfficientNet.from_pretrained(network_name, num_classes=num_classes)
    else:
        try:
            method = getattr(models, network_name)
        except AttributeError:
            raise NotImplementedError("Pytorch does not implement `{}`".format(method_name))
        network = method(pretrained=True)
    return network


def train(config):
    # Create a torch model from the network definition
    network_name = config["network"]
    num_classes = len(config["classes"])
    network = get_network(network_name, num_classes)
    model = TorchModel(
        model=network,
        name=network_name,
        out_chan=num_classes,
        gpu=config["gpu"],
        cpu_cores=config["cpu_cores"],
    )

    # set model name
    name = config["network"]
    train_folder = os.path.join(config["data_folder"], "train")

    # train the model
    print("trainfolder", train_folder)
    model.train(
        data_folder=train_folder,
        batch_size=config["batch_size"],
        l_rate=config["l_rate"],
        epochs=config["epochs"],
        config=config,
        name=name,
        heatmap_name=name + "_train_heatmap.png",
    )

    # save the model
    # now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    outname = "models/" + name + ".trch"
    torch.save(model.model, outname)


def classify(config):
    # set folder paths
    unclassified_path = config["unclassed_data_folder"]
    classified_path = config["classed_data_folder"]

    # create output folder
    if not os.path.isdir(classified_path):
        os.makedirs(classified_path)
        print("The predictions will be output to:", classified_path)
    for item in config["classes"]:
        classpath = os.path.join((classified_path), item)
        if not os.path.isdir(classpath):
            os.makedirs(classpath)

    # load the model
    print("Loading the model...")
    model = TorchModel(
        model=(config["model_path"] + ".trch"),
        out_chan=len(config["classes"]),
        gpu=config["gpu"],
        cpu_cores=config["cpu_cores"],
    )

    # load the images
    print("Loading the images...")
    image_paths = [
        os.path.join(unclassified_path, f)
        for f in os.listdir(unclassified_path)
        if os.path.isfile(os.path.join(unclassified_path, f))
    ]

    # classify the images
    print("Classifying the images...")
    classes = config["classes"]  # set classes
    resizer = Resize(config)
    # for every image in input folder
    for image_path in tqdm(image_paths):
        # load image
        image = cv2.imread(image_path)
        image_input = resizer(image)

        # get prediction
        prediction = model.predict(image_input)
        pred_class = classes[prediction[0]]

        # get file name
        if image_path.rfind("\\") == -1:
            filename = image_path[(image_path.rfind("/")) + 1 : image_path.rfind(".")]
        else:
            filename = image_path[(image_path.rfind("\\")) + 1 : image_path.rfind(".")]

        filepath = "%s/%s/%s.png" % (classified_path, pred_class, filename)
        # save image in class folder
        cv2.imwrite(filepath, image)

    print("Classification complete")
