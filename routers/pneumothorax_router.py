import io
import tensorflow as tf
from tensorflow import keras

from fastapi import APIRouter, File
from PIL import Image
from keras.preprocessing.image import img_to_array

# from TrainPneumothorax import Train
router = APIRouter()


@router.post('/predictPneumothorax')
def pneumothorax_router(image_file: bytes = File(...)):
    # model = Train().define_model()
    # model.load_weights('classifier/models/weights.h5')
    model = keras.models.load_model('classifier/models/pneumothorax.h5')
    image = Image.open(io.BytesIO(image_file))

    if image.mode != 'L':
        image = image.convert('L')

    image = image.resize((150, 150))
    image = img_to_array(image)/255.0
    image = image.reshape(1, 150, 150, 1)

    graph = tf.get_default_graph()

    with graph.as_default():
        prediction = model.predict_proba(image)

    predicted_class = 'pneumothorax' if prediction[0] > 0.5 else 'normal'

    return {'predicted_class': predicted_class,
            'pneumothorax_probability': str(prediction[0])}