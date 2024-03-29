# Script to make hdf5 files from training and test set
import numpy as np
from skimage import io, color, exposure, transform
import pandas as pd
import os
import glob
import h5py
# root directory for traing and test data
root_dir = '../../../DS-2.2/DS-2.2-DL-exercise/Datasets/gstrb_data/'
# root_dir = # REPLACE WITH ROUTE ON AWS ZIP file

NUM_CLASSES = 43
IMG_SIZE = 48


def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


def get_class_name(class_id):
    """
    Reads from csv file and returns the name of the class for the predicted
    traffic sign
    Args:
        class_id(int): predicted class id of the sign
    Returns:
        class_name(str): class name associated with class id
    """

    # create a data frame
    sign_name_df = pd.read_csv('sign_names.csv')

    sign_name_dict = {}
    for key, value in sign_name_df.values:
        sign_name_dict[key] = value

    # for key, _ in sign_name_dict:
    #     if key == class_id:
    #         return sign_name_dict[key]
    print("class id: ", class_id)
    return sign_name_dict[class_id]



def X_and_y():

    # TRAINING DATASET
    final_training_data = root_dir + 'Final_Training/Images/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(final_training_data, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs) % 1000 == 0:
                print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    # Y = np.array(labels, dtype='uint8')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    # creates h5 file with input variables X
    with h5py.File('X.h5', 'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

    # ========================================================================== #
    
    # creating test data set
    test = pd.read_csv('GT-final_test.csv', sep=';')

    X_test = []
    y_test = []
    # i = 0
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        final_test_folder = root_dir + 'Final_Test/Images/'
        # img_path = os.path.join('data/Final_Test/Images/', file_name)
        img_path = os.path.join(final_test_folder, file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id+1)

    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='uint8')

    # X test data set
    with h5py.File('X_test.h5', 'w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

    # return X_test, y_test

    return X, Y


if __name__ == '__main__':
    X_and_y()
    
