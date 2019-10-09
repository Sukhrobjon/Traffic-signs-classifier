import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# custom
from hdf5_script import preprocess_img
from skimage import io


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


def predict_model():
    model = load_model('simple_model_gtsrb.h5')
    # The default Graph being used in the current thread.
    graph = tf.get_default_graph()

    img = 'img_3.ppm'  # stop
    # preprocess the image the same way as we trained the data
    img = preprocess_img(io.imread(img))
    # reshaping to 4d to make it compatible with input shape
    x = img.reshape(1, 3, 48, 48)

    with graph.as_default():
        out = model.predict(x)

    print(out[0])
    class_id = np.argmax(out[0])
    print("class id: ", class_id)
    class_name = get_class_name(class_id)
    return (f"predicted number: {class_name}")


if __name__ == "__main__":
    print(predict_model())
