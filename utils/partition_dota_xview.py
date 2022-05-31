# partition_rareplanes_validation.py
# Dan Popp
# 10/27/21
#
# This script will partition the dota and xview testing data into a separate validation dataset used
# to validate the model during training in a much quicker manner.
import math
import os
import random
import argparse
import shutil
from tqdm import tqdm


def create_dir(path):
    """
    Creates the directory if it doesn't already exist
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root path for the Rareplanes dataset")
    parser.add_argument("--train-percent", type=float, default=0.8)
    args = parser.parse_args()

    data_path = args.data_root
    dataset_name = os.path.basename(data_path)
    imagefolder_path = os.path.join(os.path.dirname(data_path), '%s_ImageFolder' % dataset_name)
    create_dir(imagefolder_path)

    train_path = os.path.join(imagefolder_path, 'train')
    validation_path = os.path.join(imagefolder_path, 'val')
    create_dir(train_path)
    create_dir(validation_path)

    for class_dir in os.listdir(data_path):
        # Create new directory for validation class data
        full_class_dir = os.path.join(data_path, class_dir)
        full_class_dir_valid = os.path.join(validation_path, class_dir)
        full_class_dir_train = os.path.join(train_path, class_dir)
        create_dir(full_class_dir_valid)
        create_dir(full_class_dir_train)

        # Find total number of files
        file_names = [name for name in os.listdir(full_class_dir) if os.path.isfile(os.path.join(full_class_dir, name))]
        file_names.sort()

        # Take random sample of 'class_size' images to put into the validation set.
        train_num = math.floor(args.train_percent * len(file_names))
        val_num = len(file_names) - train_num
        train_indices = random.sample(range(len(file_names)), train_num)
        val_indices = []

        # Create txt file documenting validation split
        validation_text_file = os.path.join(imagefolder_path, 'train_validation_split.txt')
        with open(validation_text_file, 'a') as split_file:
            for i in range(len(file_names)):
                file = file_names[i]
                train_img = i in train_indices
                if not train_img:
                    val_indices.append(i)
                include = 1 if train_img else 0
                split_file.write('%s %d\n' % (file, include))

        for index in tqdm(train_indices):
            file_name = file_names[index]
            full_file_path = os.path.join(full_class_dir, file_name)
            # Copy file into new validation dir
            shutil.copy(full_file_path, full_class_dir_train)

        for index in tqdm(val_indices):
            file_name = file_names[index]
            full_file_path = os.path.join(full_class_dir, file_name)
            # Copy file into new validation dir
            shutil.copy(full_file_path, full_class_dir_valid)


if __name__ == '__main__':
    main()
