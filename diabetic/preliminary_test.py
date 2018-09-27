import keras
import pandas as pd
from glob import glob
import cv2
import numpy as np
from keras.models import load_model

preprocess_path = "pure_test/"
train_files = glob(os.path.join(preprocess_path, "*jpeg"))
df = pd.read_csv("trainLabels.csv")


def load_data_label(train_files, df):
    dataset = []
    labels_all=[]
    healthy = 0
    disease = 0
    for f in train_files:
        img_data=cv2.imread(f)
        basename = os.path.basename(f)
        image_id = basename.split(".")[0]
        mini_df = df[df['image'] == image_id]
        if len(mini_df) < 1:
            continue
        label = mini_df.values[0][1]

        if label == 0:
            healthy += 1
            if healthy > 220:
                healthy = 220
                continue
        else:
            disease += 1
        labels_all.append(label)
        dataset.append(img_data)
    labels_all = keras.utils.to_categorical(labels_all, num_classes=5).astype(np.float16)
    dataset = np.array(dataset).astype(np.float16)
    return dataset, labels_all, healthy, disease


if __name__ == '__main__':
    model = load_model("final_model_inv3.hdf5")
    model.summary()

    dataSet, labelSet , positive, negative = load_data_label(train_files, df)
#    divide_number = len(train_files) // 8
    data_test = dataSet
    label_test = labelSet
#    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

    score = model.evaluate(data_test, label_test, verbose=1)
    rate = 100*positive/(positive + negative)
    print("Positive rate is: " + str(round(rate,4)) + "%")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
