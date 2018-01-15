from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import glob

def prediction():
    img_width, img_height = 224, 224

    model = load_model('model.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    validation_t1 = '/media/mingrui/960EVO/workspace/deep_IDH1/modality_detect/data/validation/t1/*'
    validation_t2 = '/media/mingrui/960EVO/workspace/deep_IDH1/modality_detect/data/validation/t2/*'
    test_t2 = '/media/mingrui/DATA/datasets/201801-IDH-jpeg-binary-test/CE/*'
    test_t1 = '/media/mingrui/DATA/datasets/201801-IDH-jpeg-binary-test/T1/*'

    img_list = []

    for filename in glob.glob(test_t1):
        img = image.load_img(filename, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x *= 1.0 / x.max()
        x = np.expand_dims(x, axis=0)
        img_list.append(x)

    print(len(img_list))

    images = np.vstack(img_list)

    predictions = model.predict(images)
    print(predictions)
    prediction_average = sum(predictions) / float(len(predictions))
    print(prediction_average)

if __name__ == '__main__':
    prediction()