import pandas as pd
import numpy as np
import argparse
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.optimizers import Adamax
from keras.utils import np_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


class Train:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--epochs", type=int, default=1)
        self.args = parser.parse_args()

        K.set_learning_phase(0)
        self.data = pd.read_csv("../data/fer2013.csv", sep=",")
        self.x_train_input = []
        self.y_train = []
        self.x_test_input = []
        self.y_test = []
        self.x_train_len = 0
        self.vgg16 = None
        self.sequential_model = None
        self.merged_model = None

    def prepare_data(self):
        pandas_pixels = self.data["pixels"].tolist()
        pandas_emotions = self.data["emotion"].tolist()
        y_data_categorical = np_utils.to_categorical(pandas_emotions, 7)

        images = []
        for i in range(0, len(pandas_pixels)):
            pixels = np.array(pandas_pixels[i].split(" "), "uint8").reshape((48, 48))
            images.append(pixels)
        images = np.array(images)
        images = images.astype("float32") / 255.0

        no_of_train_examples = self.data["Usage"].value_counts()[0]

        x_train = images[0:no_of_train_examples-1]
        x_test = images[no_of_train_examples:]

        self.y_train = y_data_categorical[0:no_of_train_examples-1]
        self.y_test = y_data_categorical[no_of_train_examples:]

        self.x_train_len = int(len(x_train))
        self.x_train_input = np.empty([self.x_train_len, 48, 48, 3])
        for i, item in enumerate(self.x_train_input):
            item[:, :, 0] = x_train[i]
            item[:, :, 1] = x_train[i]
            item[:, :, 2] = x_train[i]

        self.x_test_input = np.empty([int(len(x_test)), 48, 48, 3])
        for i, item in enumerate(self.x_test_input):
            item[:, :, 0] = x_test[i]
            item[:, :, 1] = x_test[i]
            item[:, :, 2] = x_test[i]

    def build_and_train_model(self):
        # vgg 16. include_top=False so the output is the 512 and use the learned weights
        self.vgg16 = VGG16(include_top=False, input_shape=(48, 48, 3), pooling="avg", weights="imagenet")

        # get vgg16 outputs
        pic_features = self.vgg16.predict(self.x_train_input)
        x_train_feature_map = np.empty([self.x_train_len, 512])
        for pic_id, picture in enumerate(pic_features):
            x_train_feature_map[pic_id] = picture

        # build and train model
        self.sequential_model = Sequential()
        self.sequential_model.add(Dense(256, input_shape=(512,), activation="relu"))
        self.sequential_model.add(Dense(256, input_shape=(256,), activation="relu"))
        self.sequential_model.add(Dropout(0.5))
        self.sequential_model.add(Dense(128, input_shape=(256,)))
        self.sequential_model.add(Dense(7, activation="softmax"))

        self.sequential_model.compile(loss="categorical_crossentropy", optimizer=Adamax(), metrics=["accuracy"])

        # Train
        self.sequential_model.fit(x_train_feature_map, self.y_train,
                                  validation_data=(x_train_feature_map, self.y_train),
                                  epochs=self.args.epochs, batch_size=self.args.batch)

    def merge_and_evaluate_final_model(self):
        # Merge VGG and top model and create the final model
        inputs = Input(shape=(48, 48, 3))
        vg_output = self.vgg16(inputs)
        model_predictions = self.sequential_model(vg_output)
        self.merged_model = Model(inputs=inputs, outputs=model_predictions)
        self.merged_model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])

        # Evaluate on train data
        train_score = self.merged_model.evaluate(self.x_train_input, self.y_train, batch_size=self.args.batch)
        print("Model train score:", train_score)

        # Evaluate on test data
        test_score = self.merged_model.evaluate(self.x_test_input, self.y_test, batch_size=self.args.batch)
        print("Model test score:", test_score)

        print("Final model input name: ", self.merged_model.input)
        print("Final model output name: ", self.merged_model.output)

    def save_final_model(self):
        builder = saved_model_builder.SavedModelBuilder("../model/")
        signature = predict_signature_def(inputs={'images': self.merged_model.input},
                                          outputs={'scores': self.merged_model.output})

        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})
            builder.save()

    def run_train(self):
        self.prepare_data()
        self.build_and_train_model()
        self.merge_and_evaluate_final_model()
        self.save_final_model()


train = Train()
train.run_train()
