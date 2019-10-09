# Make a flask API for our DL Model

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf
from skimage import io

# CUSTOM IMPORTS
from hdf5_script import get_class_name, preprocess_img


application = app = Flask(__name__)

api = Api(app, version='1.0', title='Traffic Sign Classifier',
          description='CNN for Traffic Sign')
ns = api.namespace('Make_School DS-2.3', description='Methods')

single_parser = api.parser()
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

model = load_model('simple_model_gtsrb.h5')
graph = tf.get_default_graph()


@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload a traffic sign')
    def post(self):
        args = single_parser.parse_args()
        image_file = args.file
        # we are saving the user input
        image_file.save('img_1.jpg')
        #  now we are opening the file
        # img = Image.open('img_1.jpg')
        # image_red = img.resize((28, 28))
        # image = img_to_array(image_red)

        # IMAGE PREPROCESSING
        img = Image.open('img_1.jpg')
        # preprocess the image the same way as we trained the data
        img = preprocess_img(io.imread(img))
        # reshaping to 4d to make it compatible with input shape
        x = img.reshape(1, 3, 48, 48)
        print(x.shape)
        
        # # (1 image, gray scale)
        # # (how many, size, size, channel(scale, 1 or 3))
        # x = image.reshape(1, 28, 28, 1)
        # x = x/255


        # This is not good, because this code implies that the model will be
        # loaded each and every time a new request comes in.
        # model = load_model('my_model.h5')
        with graph.as_default():
            out = model.predict(x)

        # # print(out[0])
        # print(np.argmax(out[0]))
        # r = np.argmax(out[0])
        class_id = np.argmax(out[0])
        class_name = get_class_name(class_id)
        return (f"predicted class: {class_name}")

        # return {'prediction': str(r)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
