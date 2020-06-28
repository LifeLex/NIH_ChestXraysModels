import zipfile
import logging
import csv
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from idlelib import history
from itertools import islice
from sklearn.utils import shuffle
from keras.utils.vis_utils import plot_model


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import os
from keras.callbacks import History

""""Missing loggin functionality"""
#Custom Parameters
test_trained_model = True

class Train:
    def __init__(self):
        self.data_path = '/Volumes/My Passport/MachineLearning:AI/Datasets/NIH14ChestXray/SplittedAndLabelledImages'
        # self.model_path
        self.model: Sequential
        self.val: ImageDataGenerator
        self.train: ImageDataGenerator
        self.test: ImageDataGenerator
        self.image_size = (150,150)
        self.color_mode= 'grayscale'
        self.batch_size= 32
        self.epochs = 10

    # def take(n, iterable):
    #     "Return first n items of the iterable as a list"
    #     return Train.Convert(list(islice(iterable, n)))
    #
    #
    # def Convert(a):
    #     "Convert a list to a dict"
    #     it = iter(a)
    #     res_dct = dict(zip(it, it))
    #     return res_dct
    def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
        if type(ims[0]) is np.ndarray:
            ims = np.array(ims).astype(np.uint8)
            if (ims.shape[-1] != 3):
                ims = ims.transpose((0, 2, 3, 1))
        f = plt.figure(figsize=figsize)
        cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
        for i in range(len(ims)):
            sp = f.add_subplot(rows, cols, i + 1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            plt.imshow(ims[i], interpolation=None if interp else 'none')

    """"Function to generate the csv containing the path to the image plus the label of it"""
    def generate_csv(self):


        nfilestrain = len([name for name in os.listdir(os.path.join(self.data_path, 'train/Pneumothorax')) if
                          os.path.isfile(os.path.join(os.path.join(self.data_path, 'train/Pneumothorax'), name))])
        filenamesPneumothoraxtrain = [os.path.join(self.data_path,'train/Pneumothorax',name) for name in os.listdir(os.path.join(self.data_path, 'train/Pneumothorax')) if
                           os.path.isfile(os.path.join(os.path.join(self.data_path, 'train/Pneumothorax'), name))]
        filenamesNoFindingtrain = [os.path.join(self.data_path, 'train/No_Finding',name) for name in
                                  os.listdir(os.path.join(self.data_path, 'train/No_Finding')) if
                                  os.path.isfile(os.path.join(os.path.join(self.data_path, 'train/No_Finding'), name))]
        dictNoFindingTrain = dict.fromkeys(filenamesNoFindingtrain[:nfilestrain], 'No_Finding')
        dictPneumothoraxTrain = dict.fromkeys(filenamesPneumothoraxtrain, 'Pneumothorax')
        dictTrain = {**dictNoFindingTrain, **dictPneumothoraxTrain}

        nfilesvalidation = len([name for name in os.listdir(os.path.join(self.data_path, 'validation/Pneumothorax')) if
                          os.path.isfile(os.path.join(os.path.join(self.data_path, 'validation/Pneumothorax'), name))])
        filenamePneumothoraxvalidation = [os.path.join(self.data_path,'validation/Pneumothorax',name) for name in os.listdir(os.path.join(self.data_path, 'validation/Pneumothorax')) if
                                os.path.isfile(os.path.join(os.path.join(self.data_path, 'validation/Pneumothorax'), name))]
        filenamesNoFindingvalidation = [os.path.join(self.data_path, 'validation/No_Finding',name) for name in
                                  os.listdir(os.path.join(self.data_path, 'validation/No_Finding')) if
                                  os.path.isfile(os.path.join(os.path.join(self.data_path, 'validation/No_Finding'), name))]
        dictNoFindingValidation = dict.fromkeys(filenamesNoFindingvalidation[nfilestrain:(nfilesvalidation+nfilestrain)], 'No_Finding')
        dictPneumothoraxValidation = dict.fromkeys(filenamePneumothoraxvalidation, 'Pneumothorax')
        dictValidation = {**dictNoFindingValidation, **dictPneumothoraxValidation}

        nfilestest = len([name for name in os.listdir(os.path.join(self.data_path, 'test/Pneumothorax')) if
                          os.path.isfile(os.path.join(os.path.join(self.data_path, 'test/Pneumothorax'), name))])
        filenamesPneumothoraxtest = [os.path.join(self.data_path, 'test/Pneumothorax',name) for name in
                                     os.listdir(os.path.join(self.data_path, 'test/Pneumothorax')) if
                                     os.path.isfile(
                                         os.path.join(os.path.join(self.data_path, 'test/Pneumothorax'), name))]

        filenamesNoFindingtest = [os.path.join(self.data_path, 'test/No_Finding',name) for name in
                                  os.listdir(os.path.join(self.data_path, 'test/No_Finding')) if
                                  os.path.isfile(os.path.join(os.path.join(self.data_path, 'test/No_Finding'), name))]
        dictNoFindingTest = dict.fromkeys(filenamesNoFindingtest[nfilesvalidation:(nfilestest+nfilesvalidation)], 'No_Finding')
        dictPneumothoraxTest = dict.fromkeys(filenamesPneumothoraxtest, 'Pneumothorax')
        dictTest = {**dictNoFindingTest, **dictPneumothoraxTest}

        with open(os.path.join('csvFiles','testPneumothorax.csv'),'w') as csv_file:
            fieldnames = ['Full_Path','class']
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)
            for key , value in dictTest.items():
                writer.writerow([key,value])
        with open(os.path.join('csvFiles','trainPneumothorax.csv'),'w') as csv_file:
            fieldnames = ['Full_Path','class']
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)
            for key , value in dictTrain.items():
                writer.writerow([key,value])
        with open(os.path.join('csvFiles','validationPneumothorax.csv'),'w') as csv_file:
            fieldnames = ['Full_Path','class']
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)
            for key , value in dictValidation.items():
                writer.writerow([key,value])




    def load_data(self):
        train_df = pd.read_csv('csvFiles/trainPneumothorax.csv')
        validation_df = pd.read_csv('csvFiles/validationPneumothorax.csv')
        test_df = pd.read_csv('csvFiles/testPneumothorax.csv')
        train_df = shuffle(train_df)
        validation_df = shuffle(validation_df)
        test_df = shuffle(test_df)
        #ImageDataGenerator inherited from previous model construction, no rescale on this one because it was done previously should add it now
        """ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
        """

        train_generator = ImageDataGenerator(rescale=1 / 255.0,
                                             featurewise_center=False,  # set input mean to 0 over the dataset
                                             samplewise_center=False,  # set each sample mean to 0
                                             featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                             samplewise_std_normalization=False,  # divide each input by its std
                                             zca_whitening=False,  # apply ZCA whitening
                                             rotation_range=30,
                                             # randomly rotate images in the range (degrees, 0 to 180)
                                             zoom_range=0.2,  # Randomly zoom image
                                             width_shift_range=0.1,
                                             # randomly shift images horizontally (fraction of total width)
                                             height_shift_range=0.1,
                                             # randomly shift images vertically (fraction of total height)
                                             horizontal_flip=False,  # randomly flip images
                                             vertical_flip=False)  # randomly flip images
        test_generator = ImageDataGenerator(rescale=1 / 255.0)
        val_generator = ImageDataGenerator(rescale=1 / 255.0)
        "Need to solve UserWarning: Found 6520 invalid image filename(s) in x_col=, when removing validate filenames"
        self.train = train_generator.flow_from_dataframe(dataframe=train_df,
                                                         x_col='Full_Path',
                                                         y_col='class',
                                                         class_mode='binary',
                                                         target_size=self.image_size,
                                                         color_mode=self.color_mode,
                                                         batch_size=self.batch_size)
        print(self.train.class_indices)
        self.test= test_generator.flow_from_dataframe(dataframe=test_df,
                                                         x_col='Full_Path',
                                                         y_col='class',
                                                         class_mode='binary',
                                                         target_size=self.image_size,
                                                         color_mode=self.color_mode,
                                                         batch_size=self.batch_size)


        imgs , labels = next(self.test)
        fig, m_axs = plt.subplots(4,4, figsize= (16,16))
        for (c_x, c_y, c_ax) in zip(imgs,labels, m_axs.flatten()):
            c_ax.imshow(c_x[:, :, 0], cmap='binary', vmin=-1.5, vmax=1.5)
            if c_y > 0.5 :
                c_ax.set_title('Pneumothorax')
            else:
                c_ax.set_title('Normal')

            c_ax.axis('off')
        plt.show()
        #Train.plots(imgs, titles=labels)

        self.val = val_generator.flow_from_dataframe(dataframe=validation_df,
                                                         x_col='Full_Path',
                                                         y_col='class',
                                                         class_mode='binary',
                                                         target_size=self.image_size,
                                                         color_mode=self.color_mode,
                                                         batch_size=self.batch_size)
        #DO NOT DO DATA AUGMENTATION ON VALIDATION OR TEST DATA
        """          +-> training set ---> data augmentation --+
          |                                         |
          |                                         +-> model training --+
          |                                         |                    |
all data -+-> validation set -----------------------+                    |
          |                                                              +-> model testing
          |                                                              |
          |                                                              |
          +-> test set --------------------------------------------------+"""
        #The idea could be to do 2 functions, flow from directory one and flow from generator
        #Also plotting the data is needed
        #Also plotting the data is needed

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
        model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return model

    def train_model(self):
        self.load_data()
        steps_per_epoch = self.train.samples // self.train.batch_size
        validation_steps = self.val.samples // self.val.batch_size
        self.model = self.define_model()
        history = self.model.fit_generator(self.train,
                                 epochs=self.epochs,
                                 validation_data=self.val,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)
        self.model.save('classifier/models/pneumothorax.h5')

        print(history.history.keys())
        epochs = [i for i in range(self.epochs)]
        fig, ax = plt.subplots(1, 2)
        train_acc = history.history['accuracy']
        train_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        fig.set_size_inches(20, 10)
        ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
        ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
        ax[0].set_title('Training & Validation Accuracy')
        ax[0].legend()
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
        ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
        ax[1].set_title('Testing Accuracy & Loss')
        ax[1].legend()
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Training & Validation Loss")
        plt.show()


    def deploy_model(self):
        self.train_model()

        loss, acc = self.model.evaluate_generator(self.test,
                                                  steps=self.test.samples // self.test.batch_size,
                                                  verbose=1)
        print('Model has been trained with loss, accuracy of {}, {}'.format(loss, acc))

    def test_model(self):
        self.load_data()
        print(self.test.samples)
        step_size_test= self.test.samples / self.test.batch_size
        loaded_model = keras.models.load_model('classifier/models/pneumothorax.h5')
        predictions = loaded_model.predict_generator(self.test,
                                                 steps=step_size_test,
                                                 verbose=1)



        test_predict= (predictions >0.5).astype(np.int8)
        print(test_predict)
        print(self.test.labels)
        conf_matrix = confusion_matrix(y_true=self.test.labels, y_pred=test_predict)
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', cbar=False, square=True, xticklabels=['No_Finding', 'Pneumothorax'],
                    yticklabels=['No_Finding', 'Pneumothorax'])
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        # print(predictions_classes)

if __name__ == '__main__':
    train_model = Train()
    train_model.generate_csv()
    if not test_trained_model:
        train_model.deploy_model()
    else:
        train_model.test_model()