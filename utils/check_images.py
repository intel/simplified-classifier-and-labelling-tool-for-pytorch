import os
import cv2
import hjson
import imghdr
import glob
from PIL import Image


def main():
    delete = False
    os.chdir("..")
    with open("config.hjson") as config_file:
        config = hjson.load(config_file)

    file_ending = config["file_ending"]
    images = find("*." + file_ending)

    for image in images:

        # look for the invalids
        try:
            img = Image.open(image)
            img.verify()
        except Exception:
            print("Invalid image: " + image)
            if delete:
                os.remove(image)
            continue
        # look for empties
        imtype = imghdr.what(image)
        if imtype is None:
            print("Empty image: " + image)
            if delete:
                os.remove(image)
        else:
            # and incorrectly packaged
            imtype = imtype.replace("e", "")  # because it calls them jpegs
            ending = image.split(".")[-1]
            if imtype != ending:
                print("Wrong Wrapper:" + image + " " + imtype + " != " + ending)
                if delete:
                    os.remove(image)

    # check_duplicates(images, delete)


def find(regex, folder="./"):
    found = []
    for filename in glob.iglob(folder + "**/" + regex, recursive=True):
        found.append(filename)
    return found


def check_duplicates(imagePaths, delete):
    hashes = {}

    # loop over our image paths
    for imagePath in imagePaths:
        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        h = dhash(image)

        # grab all image paths with that hash, add the current image
        # path to it, and store the list back in the hashes dictionary
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p

    # loop over the image hashes
    for (h, hashedPaths) in hashes.items():
        # check to see if there is more than one image with the same hash
        if len(hashedPaths) > 1:
            # check to see if this is a dry run
            if delete:
                # loop over all image paths with the same hash *except*
                # for the first image in the list (since we want to keep
                # one, and only one, of the duplicate images)
                for p in hashedPaths[1:]:
                    os.remove(p)
            else:
                # initialize a montage to store all images with the same
                # hash
                montage = None

                # loop over all image paths with the same hash
                for p in hashedPaths:
                    # load the input image and resize it to a fixed width
                    # and height
                    image = cv2.imread(p)
                    image = cv2.resize(image, (150, 150))

                    # if our montage is None, initialize it
                    if montage is None:
                        montage = image

                    # otherwise, horizontally stack the images
                    else:
                        montage = np.hstack([montage, image])

                # show the montage for the hash
                print("[INFO] hash: {}".format(h))
                cv2.imshow("Montage", montage)
                cv2.waitKey(0)


def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


if __name__ == "__main__":
    main()
