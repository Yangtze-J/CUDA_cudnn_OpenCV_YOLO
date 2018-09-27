import keras
import pandas as pd
from glob import glob
import os
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications import Xception
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from sklearn import metrics

preprocess_path = "../input/512/"
train_files = glob(os.path.join(preprocess_path, "train/*jpeg"))
df = pd.read_csv("../input/trainLabels.csv")

img_height = 299
img_width = 299
batch_size = 24


def load_data_label(train_files, df):
    dataset = []
    labels_all=[]
    for f in train_files:
        img_data=cv.imread(f)
        dataset.append(img_data)

        basename = os.path.basename(f)
        # os.path.basename() return the last part of the filename divided by '\' or '/'
        # 13_left.jpeg
        image_id = basename.split(".")[0]
        # 13_left
        mini_df = df[df['image'] == image_id]
        # print(mini_df)
        #       image      level
        #    0  13_left      0
        if len(mini_df) < 1:
            continue
        label = mini_df.values[0][1]
        # mini_df.values : [['13_left' 0]]    mini_df.values[0] : ['13_left' 0]  mini_df.values[0][1] : 0
        labels_all.append(label)
    labels_all = keras.utils.to_categorical(labels_all, num_classes=5).astype(np.float16)
    dataset = np.array(dataset).astype(np.float16)
    return dataset, labels_all


img_channels = 3
img_dim = (img_height, img_width, img_channels)


def inceptionv3(img_dim=img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(5, activation='softmax')(x)
    models = Model(input_tensor, output)
    return models


def flatten_list(l):
    return [item for sublist in l for item in sublist]


if __name__ == '__main__':
    # model = darknet_based('darknet53.cfg', 'darknet53_weights.h5')
    # model.summary()

    model = inceptionv3(img_dim)
    model.summary()

    dataSet, labelSet = load_data_label(train_files, df)

    divide_number = len(train_files) // 10
    data_test = dataSet[:divide_number]
    label_test = labelSet[:divide_number]

    data_eval = dataSet[divide_number : 2*divide_number]
    label_eval = labelSet[divide_number : 2*divide_number]

    data_train = dataSet[2*divide_number:]
    label_train = labelSet[2*divide_number:]
    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"


# use multigpu 4
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics = ['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint("darkNet_sgd"+".hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
    ]

    history = parallel_model.fit(data_train,label_train,batch_size=batch_size,epochs=40,verbose=1,callbacks=callbacks,
                                 validation_data=(data_eval, label_eval))

    score = model.evaluate(data_test, label_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("final_model.hdf5")
