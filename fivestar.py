import os
import sys
import hjson
import pytorch_classifier as classifier
from argparse import ArgumentParser


def main():
    args = build_argparser().parse_args()
    config, classifier = load_config(("config.hjson"))

    if args.task == "train":
        print("training")
        classifier.train(config)
    if args.task == "test":
        classifier.test(config)
    if args.task == "classify":
        classifier.classify(config)


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("task", type=str, help="the task to be carried out: train, test, or classify")
    return parser


def load_config(config_name):

    # load the config file
    if os.path.exists(config_name):
        with open(config_name) as config_file:
            config = hjson.load(config_file)
    else:
        print(
            "You must specify a config.hjson file, there is an exampleconfig.hjson in the root folder that you can use as a template."
        )
        exit()
    # set gpu number if working on the server
    if "cuda_visible_devices" in config.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    if config["gpu"] is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return config, classifier


if __name__ == "__main__":
    main()
