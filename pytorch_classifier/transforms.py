## Subclassed transforms to allow n-dimensional transforms vs. PIL style transforms of 1-4D
#   @author anton.shmatov@intel.com
#   @date 8/10/2019
#

import numpy as np
import random

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from cv2 import blur, calcHist, Canny as CVCanny
from pytorch_classifier.aspectawarepreprocessor import AspectAwarePreprocessor as AAP


class RandomRotation(transforms.RandomRotation):
    def __call__(self, img):
        """
        Args:
            img (Numpy): Image to be rotated.

        Returns:
            Numpy Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        rotated = [
            np.array(
                F.rotate(
                    Image.fromarray(img[:, :, dim]),
                    angle,
                    self.resample,
                    self.expand,
                    self.center,
                )
            )
            for dim in range(img.shape[2])
        ]

        return np.dstack(rotated)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img):
        """
        Args:
            img (Numpy Image): Image to be flipped.

        Returns:
            Numpy Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = [
                np.array(F.hflip(Image.fromarray(img[:, :, dim])))
                for dim in range(img.shape[2])
            ]
            return np.dstack(img)
        return img


class HighPass:
    """
    Gets the high frequency features of the image
    """

    def __init__(self, kernel=(5, 5)):
        self.kernel = kernel

    def __call__(self, img):
        blurred = blur(img, self.kernel)

        img = img.astype("float32") - blurred
        img = img - img.min()
        img = img / img.max()

        return img


class ModalDelete:
    """
    Deletes the modal components of the image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        img = img.copy()
        hist = calcHist([img], [0], None, [256], [0, 256])

        max_indices = np.where(hist > hist.mean())[0]
        mean = int(max_indices.mean())

        for index in max_indices:
            img[img == index] = mean

        return img


class Canny:
    """
    Performs canny detection by finding the min and max of modal components
    """

    def __init__(self):
        pass

    def __call__(self, img):
        img_out = img.copy()

        for dim in range(img.shape[2]):
            hist = calcHist([img_out[:, :, dim]], [0], None, [256], [0, 256])

            max_indices = np.where(hist > hist.mean())[0]

            thresh1 = max_indices.min()
            thresh2 = max_indices.max()

            img_out[:, :, dim] = CVCanny(img_out[:, :, dim], thresh1, thresh2)

        return img_out


class Resize:
    """
    Resizes the image according to the specified config
    """

    def __init__(self, config):
        self._aap = AAP(int(config["image_width"]), int(config["image_height"]))

    def __call__(self, img):
        """
        Use the AAP to resize the image
        """
        return self._aap.preprocess(img)
