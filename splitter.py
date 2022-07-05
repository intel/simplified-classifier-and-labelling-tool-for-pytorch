import hjson
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split


def main():
    # hoppping back a folder for relative paths
    if os.path.exists("config.hjson"):
        with open("config.hjson") as config_file:
            config = hjson.load(config_file)
    else:
        print(
            "You must specify a config.hjson file, there is an exampleconfig.hjson in the root folder that you can use as a template."
        )
        exit()

    dataset = config["data_folder"]
    traindir = os.path.join(dataset, "train")
    testdir = os.path.join(dataset, "test")
    split = config["train_test_split"]
    subfolders = [f.path for f in os.scandir(dataset) if f.is_dir()]

    if not (os.path.isdir(traindir)):
        os.makedirs(traindir)
    if not (os.path.isdir(testdir)):
        os.makedirs(testdir)

    for subfolder in subfolders:
        if (subfolder.find("train") == -1) and (subfolder.find("test") == -1):
            splitdir(subfolder, split, traindir, testdir)


def splitdir(src, split, traindir, testdir):
    src = os.path.join(src, "")  # add trailing slash if missing
    X = y = os.listdir(src)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=0)
    # need to replace this for windows systems
    src = src.replace("\\", "/")
    label = src.split("/")[-2]
    if not (os.path.isdir(os.path.join(traindir, label))):
        os.makedirs(os.path.join(traindir, label))
    if not (os.path.isdir(os.path.join(testdir, label))):
        os.makedirs(os.path.join(testdir, label))
    for x in X_train:
        print("copying", src + x, os.path.join(traindir, label, x))
        copyfile(src + x, os.path.join(traindir, label, x))
    for x in X_test:
        print("copying", src + x, os.path.join(testdir, label, x))
        copyfile(src + x, os.path.join(testdir, label, x))


if __name__ == "__main__":
    main()
