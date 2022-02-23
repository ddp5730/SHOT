# cub200_resolutions.py
# Dan Popp
# 1/21/21
#
# The purpose of this file is to determine the resolutions of the CUB200 dataset
import argparse
import os

import PIL


def check_image_resolutions(path):
    """
    This function will check the image resolutions for the given dataset.  It will provide the average, min, and max
    resolutions.
    """

    cum_height = 0
    cum_width = 0
    num_images = 0
    max_height = 0
    max_width = 0
    min_height = float('inf')
    min_width = float('inf')

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            abs_path = os.path.join(root, name)
            img = PIL.Image.open(abs_path)
            wid, hgt = img.size

            cum_width += wid
            cum_height += hgt
            num_images += 1

            if wid < min_width:
                min_width = wid
            if wid > max_width:
                max_width = wid
            if hgt < min_height:
                min_height = hgt
            if hgt > max_height:
                max_height = hgt

    average_height = float(cum_height) / num_images
    average_width = float(cum_width) / num_images

    return average_height, average_width, min_height, max_height, min_width, max_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='The file path to the CUB200 dataset directory')

    args = parser.parse_args()

    path = args.dataset_path

    average_height, average_width, min_height, max_height, min_width, max_width = check_image_resolutions(path)
    print('Average Resolution: (%f, %f)' % (average_height, average_width))
    print('Min Resolution: (%f, %f)' % (min_height, min_width))
    print('Max Resolution: (%f, %f)' % (max_height, max_width))


if __name__ == '__main__':
    main()
