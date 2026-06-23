'''
學長的程式 - 第一篇論文(青少年)
'''
import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Reshape
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.model_selection import train_test_split

from utils import f1_m, precision_m, recall_m
from tensorflow.python.keras.utils.np_utils import to_categorical

# No scientific notations
np.set_printoptions(suppress=True)
tf.get_logger().setLevel('ERROR')

# Fix random seed
random.seed(12)
np.random.seed(34)
tf.random.set_seed(56)

def fix_None(data, fix:bool):
    if fix: # fix none and nan
        data = np.array(data, dtype=float)
        nan_indices = np.isnan(data).any(axis=1)
        data[nan_indices] = 0
        return data
    else:   # only check
        print(((data == None).any()))
        return data


def dense_layer(dim):
    model = Sequential()
    model.add(Dense(input_dim=dim, units=dim, activation='relu'))
    return model
def lstm_layer(dim):
    model = Sequential()
    model.add(LSTM(units=dim, input_shape=(None, dim)))
    return model

def three_fusion_model(category):
    """
    Return model for feature-level fusion with 3 modal features
    """
    text = dense_layer(768)
    audio = dense_layer(193)
    face = lstm_layer(256)

    combinedInput = concatenate([text.output, audio.output, face.output])

    x = Dense(units=512, activation="relu", kernel_regularizer='l2')(combinedInput)
    x = Dropout(rate=.2)(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(rate=.2)(x)
    x = Dense(units=category, activation="softmax")(x)

    model = Model(inputs=[text.input, audio.input, face.input], outputs=x)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', f1_m, precision_m, recall_m]
    )

    return model
def four_fusion_model(category):
    """
    Return model for feature-level fusion with 4 modal features
    """
    text = dense_layer(768)
    audio = dense_layer(193)
    face = lstm_layer(256)
    hrv = dense_layer(23)


    combinedInput = concatenate([text.output, audio.output, face.output, hrv.output])

    x = Dense(units=512, activation="relu", kernel_regularizer='l2')(combinedInput)
    x = Dropout(rate=.2)(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(rate=.2)(x)
    x = Dense(units=category, activation="softmax")(x)

    model = Model(inputs=[text.input, audio.input, face.input, hrv.input], outputs=x)
    
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', f1_m, precision_m, recall_m]
    )

    return model



def five_fusion_model(category):
    """
    Return model for feature-level fusion with 5 modal features
    """
    text = dense_layer(768)
    audio = dense_layer(193)
    face = lstm_layer(256)
    hrv = dense_layer(23)
    eye_movement = dense_layer(7)

    combinedInput = concatenate([text.output, audio.output, face.output, hrv.output, eye_movement.output])

    x = Dense(units=512, activation="relu", kernel_regularizer='l2')(combinedInput)
    x = Dropout(rate=.2)(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(rate=.2)(x)
    x = Dense(units=category, activation="softmax")(x)

    model = Model(inputs=[text.input, audio.input, face.input, hrv.input, eye_movement.input], outputs=x)
    
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', f1_m, precision_m, recall_m]
    )

    return model
    

def three_modal_four_class(EPOCH):
    """
    Feature-level fusion with concatenation of 3-modal features: text, audio, face
    Classify ["Mild", "Moderate", "Severe", "Manic"]
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 4
    SUBJECT = 150
    TEST_SIZE = 0.25
    LABEL = ["Mild", "Moderate", "Severe", "Manic"]
    ROOT = "three_modal_four_class"
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load features
    x_text = np.load(f"preprocess/x_split_text_{SUBJECT}.npy")
    x_audio = np.load(f"preprocess/x_split_audio_{SUBJECT}.npy")
    
    face = np.load(f"preprocess/x_split_face_{SUBJECT}.npy", allow_pickle=True)
    x_face = []
    for xs in face:
        temp = []
        for x in xs:
            temp.append(x)
        
        x_face.append(np.array(temp))
    x_face = np.array(x_face)

    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_face = np.nan_to_num(x_face)

    # Load labels
    label = np.load(f"preprocess/y_split_{SUBJECT}.npy")

    # Count unique values
    (unique, counts) = np.unique(label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    # One hot encoding
    # y = np_utils.to_categorical(label)
    y = np.zeros((label.size, label.max()))
    y[np.arange(label.size), label-1] = 1
    # print(y)

    # Model
    model = three_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    
    for train_index, test_index in rs.split(x_text):
        print(f"Train set:{train_index.shape}, Test set: {test_index.shape}")
        
        # Train
        for e in range(EPOCH):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                
                label = np.expand_dims(y[index], axis=0)
                
                model.fit(
                    x=[text, audio, face],
                    y=label
                )

        # Evaluate
        confusion_matrix = np.zeros((CATEGORY, CATEGORY))
        
        for index in test_index:
            text = np.expand_dims(x_text[index], axis=0)
            audio = np.expand_dims(x_audio[index], axis=0)
            face = np.expand_dims(x_face[index], axis=0)
            
            label = np.expand_dims(y[index], axis=0)
            
            predict = np.argmax(model.predict([text, audio, face]), axis=1)
            confusion_matrix += tf.math.confusion_matrix(np.argmax(label, axis=1), predict, num_classes=CATEGORY)
    
    # Create directory for saving confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"

    # Plot confusion matrix
    confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
    print(confusion_matrix)

    # Confusion matrix with actual number of data
    cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # plt.title("Confusion Matrix")
    # plt.ylabel("Ground Truth")
    # plt.xlabel("Predicted Value")
    img_path = f"{CM_PATH}/confusion_matrix.png"
    plt.savefig(img_path)
    print(f"{img_path} has been saved")

    # Confusion matrix with percentage of data
    uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
    cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
    plt.clf() # clear canvas
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1) # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    img_path = f"{CM_PATH}/confusion_matrix_uniformed.png"
    plt.savefig(img_path)
    print(f"{img_path} has been saved")

    # Save model
    model_name = f"{MODEL_PATH}/{ROOT}.h5"
    model.save(model_name)
    print(f"{model_name} has been saved")
    # Accuracy: (335+531+524+276)/(335+531+524+276+14+35+5+25+79+8+24+60+7+9+23+36) = 0.8367654445

def five_modal_four_class(EPOCH):
    """
    Feature-level fusion with concatenation of 5-modal features: text, audio, face, hrv, and eye-movement
    Classify ["Mild", "Moderate", "Severe", "Manic"]
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 4
    SUBJECT = 150
    TEST_SIZE = 0.25
    LABEL = ["Mild", "Moderate", "Severe", "Manic"]
    ROOT = "five_modal_four_class"
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load features
    x_text = np.load(f"preprocess/x_75%_split_text.npy")
    x_audio = np.load(f"preprocess/x_75%_split_audio.npy")
    x_hrv = np.load(f"preprocess/x_75%_split_hrv.npy")
    x_eye = np.load(f"preprocess/x_75%_split_eye.npy")
    
    face = np.load(f"preprocess/x_75%_split_face.npy", allow_pickle=True)

    x_face = []
    for xs in face:
        temp = []
        for x in xs:
            temp.append(x)
        
        x_face.append(np.array(temp))
    x_face = np.array(x_face)

    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_hrv = np.nan_to_num(x_hrv)
    x_eye = np.nan_to_num(x_eye)
    x_face = np.nan_to_num(x_face)

    # Load labels
    label = np.load(f"preprocess/y_75%_split_train.npy")
    # label = np.load(f"preprocess/y_split_{SUBJECT}.npy")

    # Count unique values
    (unique, counts) = np.unique(label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    # One hot encoding
    y = np.zeros((label.size, label.max()))
    y[np.arange(label.size), label-1] = 1
    
    # Model
    model = five_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    
    for train_index, test_index in rs.split(x_text):
        print(f"Train set:{train_index.shape}, Test set: {test_index.shape}")
        
        # Train
        for e in range(EPOCH):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                hrv = np.expand_dims(x_hrv[index], axis=0)
                eye_movement = np.expand_dims(x_eye[index], axis=0)
                
                label = np.expand_dims(y[index], axis=0)
                # print([text, audio, face, hrv, eye_movement])
                
                model.fit(
                    x=[text, audio, face, hrv, eye_movement],
                    y=label
                )

                ## Get output of certain layer
                # inter_output_model = Model(model.input, model.get_layer(index = 10).output )
                # inter_output = inter_output_model.predict([text, audio, face, hrv, eye_movement])
                # print(inter_output)

                ## Get output of all layers
                # outputs = []
                # for layer in model.layers:
                #     keras_function = K.function([model.input], [layer.output])
                #     outputs.append(keras_function([[text, audio, face, hrv, eye_movement], 1]))
                
                # for index, output in enumerate(outputs):
                #     print(index, output)

        # Evaluate
        confusion_matrix = np.zeros((CATEGORY, CATEGORY))
        
        for index in test_index:
            text = np.expand_dims(x_text[index], axis=0)
            audio = np.expand_dims(x_audio[index], axis=0)
            face = np.expand_dims(x_face[index], axis=0)
            hrv = np.expand_dims(x_hrv[index], axis=0)
            eye_movement = np.expand_dims(x_eye[index], axis=0)
            
            label = np.expand_dims(y[index], axis=0)
            
            predict = np.argmax(model.predict([text, audio, face, hrv, eye_movement]), axis=1)
            confusion_matrix += tf.math.confusion_matrix(np.argmax(label, axis=1), predict, num_classes=CATEGORY)

    # Plot confusion matrix
    confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
    print(confusion_matrix)

    # Confusion matrix with actual number of data
    cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # plt.title("Confusion Matrix")
    # plt.ylabel("Ground Truth")
    # plt.xlabel("Predicted Value")
    img_path = f"{CM_PATH}/confusion_matrix.png"
    plt.savefig(img_path)
    print(f"{img_path} has been saved")

    # Confusion matrix with percentage of data
    uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
    cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
    plt.clf() # clear canvas
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1) # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    img_path = f"{CM_PATH}/confusion_matrix_uniformed.png"
    plt.savefig(img_path)
    print(f"{img_path} has been saved")

    # Save model
    model_name = f"{MODEL_PATH}/{ROOT}.h5"
    model.save(model_name)
    print(f"{model_name} has been saved")
    # Accuracy: (359+584+545+323)/(359+584+545+323+13+12+5+11+41+7+21+28+21+3+7+11) = 0.90959316926

def three_modal_five_class(EPOCH):
    """
    Feature-level fusion with concatenation of 3-modal features: text, audio, face
    Classify 5 class: mild, moderate, severe, manic, normal
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 5
    TEST_SIZE = 0.25
    LABEL = ["Mild", "Moderate", "Severe", "Manic", "Normal"]
    ROOT = "three_modal_five_class"
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"

    # Create directory for saving model and confusion matrix
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load data
    x_text = np.concatenate((np.load(f"preprocess/x_75%_split_text.npy"), np.load(f"preprocess/x_75%_normal_text.npy"), np.load(f"preprocess/x_25%_split_text.npy"), np.load(f"preprocess/x_25%_normal_text.npy")), axis=0)
    x_audio = np.concatenate((np.load(f"preprocess/x_75%_split_audio.npy"), np.load(f"preprocess/x_75%_normal_audio.npy"), np.load(f"preprocess/x_25%_split_audio.npy"), np.load(f"preprocess/x_25%_normal_audio.npy")), axis=0)
    
    face = np.concatenate((np.load(f"preprocess/x_75%_split_face.npy", allow_pickle=True), np.load(f"preprocess/x_75%_normal_face.npy", allow_pickle=True), np.load(f"preprocess/x_25%_split_face.npy", allow_pickle=True), np.load(f"preprocess/x_25%_normal_face.npy", allow_pickle=True)), axis=0)
    x_face = []
    for xs in face:
        temp = []
        for x in xs:
            temp.append(x)
        
        x_face.append(np.array(temp))
    x_face = np.array(x_face)

    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_face = np.nan_to_num(x_face)

    # Load labels
    label = np.concatenate((np.load(f"preprocess/y_75%_split_train.npy"), np.load(f"preprocess/y_75%_normal.npy"), np.load(f"preprocess/y_25%_split_test.npy"), np.load(f"preprocess/y_25%_normal.npy")), axis=0)

    # Count unique values
    (unique, counts) = np.unique(label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    # One hot encoding
    y = np.zeros((label.size, label.max()))
    y[np.arange(label.size), label-1] = 1

    # Model
    model = three_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    
    for train_index, test_index in rs.split(x_text):
        for e in range(1, EPOCH+1):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                
                y_true = np.expand_dims(y[index], axis=0)
                
                model.fit(
                    x=[text, audio, face],
                    y=y_true
                )

            # Evaluate
            if (e != 0) and (e%5 == 0):
                # init
                confusion_matrix = np.zeros((CATEGORY, CATEGORY))

                for index in test_index:
                    text = np.expand_dims(x_text[index], axis=0)
                    audio = np.expand_dims(x_audio[index], axis=0)
                    face = np.expand_dims(x_face[index], axis=0)
                    
                    y_true = np.argmax(np.expand_dims(y[index], axis=0), axis=1)
                    y_pred = np.argmax(model.predict([text, audio, face]), axis=1)
                    confusion_matrix += tf.math.confusion_matrix(y_true, y_pred, num_classes=CATEGORY)

                # Plot confusion matrix
                confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
                print(confusion_matrix)

                # Confusion matrix with actual number of data
                cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                # plt.title("Confusion Matrix")
                # plt.ylabel("Ground Truth")
                # plt.xlabel("Predicted Value")
                img_path = f"{CM_PATH}/epoch{e}.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Confusion matrix with percentage of data
                uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
                cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
                plt.clf() # clear canvas
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1) # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                img_path = f"{CM_PATH}/epoch{e}_uniformed.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Save model
                model_name = f"{MODEL_PATH}/epoch{e}.h5"
                model.save(model_name)
                print(f"{model_name} has been saved")

def five_modal_five_class(EPOCH):
    """
    Feature-level fusion with concatenation of 5-modal features: text, audio, face, hrv, and eye-movement
    Classify 5 class: mild, moderate, severe, manic, normal
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 5
    TEST_SIZE = 0.25
    LABEL = ["Mild", "Moderate", "Severe", "Manic", "Normal"]
    ROOT = "five_modal_five_class"
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"

    # Create directory for saving model and confusion matrix
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load data
    x_text = np.concatenate((np.load(f"preprocess/x_75%_split_text.npy"), np.load(f"preprocess/x_75%_normal_text.npy"), np.load(f"preprocess/x_25%_split_text.npy"), np.load(f"preprocess/x_25%_normal_text.npy")), axis=0)
    x_audio = np.concatenate((np.load(f"preprocess/x_75%_split_audio.npy"), np.load(f"preprocess/x_75%_normal_audio.npy"), np.load(f"preprocess/x_25%_split_audio.npy"), np.load(f"preprocess/x_25%_normal_audio.npy")), axis=0)
    x_hrv = np.concatenate((np.load(f"preprocess/x_75%_split_hrv.npy"), np.load(f"preprocess/x_75%_normal_hrv.npy"), np.load(f"preprocess/x_25%_split_hrv.npy"), np.load(f"preprocess/x_25%_normal_hrv.npy")), axis=0)
    x_eye = np.concatenate((np.load(f"preprocess/x_75%_split_eye.npy"), np.load(f"preprocess/x_75%_normal_eye.npy"), np.load(f"preprocess/x_25%_split_eye.npy"), np.load(f"preprocess/x_25%_normal_eye.npy")), axis=0)
    
    face = np.concatenate((np.load(f"preprocess/x_75%_split_face.npy", allow_pickle=True), np.load(f"preprocess/x_75%_normal_face.npy", allow_pickle=True), np.load(f"preprocess/x_25%_split_face.npy", allow_pickle=True), np.load(f"preprocess/x_25%_normal_face.npy", allow_pickle=True)), axis=0)
    x_face = []
    for xs in face:
        temp = []
        for x in xs:
            temp.append(x)
        
        x_face.append(np.array(temp))
    x_face = np.array(x_face)

    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_hrv = np.nan_to_num(x_hrv)
    x_eye = np.nan_to_num(x_eye)
    x_face = np.nan_to_num(x_face)

    # Load labels
    label = np.concatenate((np.load(f"preprocess/y_75%_split_train.npy"), np.load(f"preprocess/y_75%_normal.npy"), np.load(f"preprocess/y_25%_split_test.npy"), np.load(f"preprocess/y_25%_normal.npy")), axis=0)

    # Count unique values
    (unique, counts) = np.unique(label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    # One hot encoding
    y = np.zeros((label.size, label.max()))
    y[np.arange(label.size), label-1] = 1
    
    print("x_text shape:", x_text.shape)
    print("x_audio shape:", x_audio.shape)
    print("x_hrv shape:", x_hrv.shape)
    print("x_eye shape:", x_eye.shape)
    print("x_face shape:", x_face.shape)
    print("y shape:", y.shape)

    # Model
    model = five_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    
    for train_index, test_index in rs.split(x_text):
        for e in range(1, EPOCH+1):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                hrv = np.expand_dims(x_hrv[index], axis=0)
                eye_movement = np.expand_dims(x_eye[index], axis=0)
                
                y_true = np.expand_dims(y[index], axis=0)
                
                model.fit(
                    x=[text, audio, face, hrv, eye_movement],
                    y=y_true
                )

            # Evaluate
            if (e != 0) and (e%5 == 0):
                # init
                confusion_matrix = np.zeros((CATEGORY, CATEGORY))

                for index in test_index:
                    text = np.expand_dims(x_text[index], axis=0)
                    audio = np.expand_dims(x_audio[index], axis=0)
                    face = np.expand_dims(x_face[index], axis=0)
                    hrv = np.expand_dims(x_hrv[index], axis=0)
                    eye_movement = np.expand_dims(x_eye[index], axis=0)
                    
                    y_true = np.argmax(np.expand_dims(y[index], axis=0), axis=1)
                    y_pred = np.argmax(model.predict([text, audio, face, hrv, eye_movement]), axis=1)
                    confusion_matrix += tf.math.confusion_matrix(y_true, y_pred, num_classes=CATEGORY)

                # Plot confusion matrix
                confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
                print(confusion_matrix)

                # Confusion matrix with actual number of data
                cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                # plt.title("Confusion Matrix")
                # plt.ylabel("Ground Truth")
                # plt.xlabel("Predicted Value")
                img_path = f"{CM_PATH}/epoch{e}.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Confusion matrix with percentage of data
                uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
                cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
                plt.clf() # clear canvas
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1) # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                img_path = f"{CM_PATH}/epoch{e}_uniformed.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Save model
                model_name = f"{MODEL_PATH}/epoch{e}.h5"
                model.save(model_name)
                print(f"{model_name} has been saved")

def four_modal_five_class(EPOCH):
    """
    Feature-level fusion with concatenation of 4-modal features: text, audio, face, and hrv
    Classify 5 class: mild, moderate, severe, manic, and normal
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 5
    TEST_SIZE = 0.25
    LABEL = ["Mild", "Moderate", "Severe", "Manic", "Normal"]
    ROOT = "four_modal_five_class"
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"

    # Create directory for saving model and confusion matrix
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load data
    x_text = np.concatenate((np.load(f"preprocess/x_75%_split_text.npy"), np.load(f"preprocess/x_75%_normal_text.npy"), np.load(f"preprocess/x_25%_split_text.npy"), np.load(f"preprocess/x_25%_normal_text.npy")), axis=0)
    x_audio = np.concatenate((np.load(f"preprocess/x_75%_split_audio.npy"), np.load(f"preprocess/x_75%_normal_audio.npy"), np.load(f"preprocess/x_25%_split_audio.npy"), np.load(f"preprocess/x_25%_normal_audio.npy")), axis=0)
    x_hrv = np.concatenate((np.load(f"preprocess/x_75%_split_hrv.npy"), np.load(f"preprocess/x_75%_normal_hrv.npy"), np.load(f"preprocess/x_25%_split_hrv.npy"), np.load(f"preprocess/x_25%_normal_hrv.npy")), axis=0)

    face = np.concatenate((np.load(f"preprocess/x_75%_split_face.npy", allow_pickle=True), np.load(f"preprocess/x_75%_normal_face.npy", allow_pickle=True), np.load(f"preprocess/x_25%_split_face.npy", allow_pickle=True), np.load(f"preprocess/x_25%_normal_face.npy", allow_pickle=True)), axis=0)
    x_face = []
    for xs in face:
        temp = []
        for x in xs:
            temp.append(x)
        
        x_face.append(np.array(temp))
    x_face = np.array(x_face)

    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_hrv = np.nan_to_num(x_hrv)
    x_face = np.nan_to_num(x_face)

    # Load labels
    label = np.concatenate((np.load(f"preprocess/y_75%_split_train.npy"), np.load(f"preprocess/y_75%_normal.npy"), np.load(f"preprocess/y_25%_split_test.npy"), np.load(f"preprocess/y_25%_normal.npy")), axis=0)

    # Count unique values
    (unique, counts) = np.unique(label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    # One hot encoding
    y = np.zeros((label.size, label.max()))
    y[np.arange(label.size), label-1] = 1

    # Model
    model = four_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    
    for train_index, test_index in rs.split(x_text):
        for e in range(1, EPOCH+1):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                hrv = np.expand_dims(x_hrv[index], axis=0)
                
                y_true = np.expand_dims(y[index], axis=0)
                

                model.fit(
                    x=[text, audio, face, hrv],
                    y=y_true
                )

            # Evaluate
            if (e != 0) and (e%5 == 0):
                # init
                confusion_matrix = np.zeros((CATEGORY, CATEGORY))

                for index in test_index:
                    text = np.expand_dims(x_text[index], axis=0)
                    audio = np.expand_dims(x_audio[index], axis=0)
                    face = np.expand_dims(x_face[index], axis=0)
                    hrv = np.expand_dims(x_hrv[index], axis=0)
                    
                    y_true = np.argmax(np.expand_dims(y[index], axis=0), axis=1)
                    y_pred = np.argmax(model.predict([text, audio, face, hrv]), axis=1)
                    confusion_matrix += tf.math.confusion_matrix(y_true, y_pred, num_classes=CATEGORY)

                # Plot confusion matrix
                confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
                print(confusion_matrix)

                # Confusion matrix with actual number of data
                cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                # plt.title("Confusion Matrix")
                # plt.ylabel("Ground Truth")
                # plt.xlabel("Predicted Value")
                img_path = f"{CM_PATH}/epoch{e}.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Confusion matrix with percentage of data
                uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
                cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
                plt.clf() # clear canvas
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1) # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                img_path = f"{CM_PATH}/epoch{e}_uniformed.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Save model
                model_name = f"{MODEL_PATH}/epoch{e}.h5"
                model.save(model_name)
                print(f"{model_name} has been saved")



# New Data struture
def Dataset(filepath: str , catagory):
        
    if filepath != "":
        
        audio, eye, text, hrv, face = [], [], [], [], []
        label = []

        for sender_id in os.listdir(filepath):
            # 取得子資料夾路徑
            sender_id_path = os.path.join(filepath, sender_id)

            # 忽略非資料夾檔案
            if not os.path.isdir(sender_id_path) or sender_id.startswith('__'):
                continue
            
            new_audio = np.load( os.path.join(sender_id_path, "x_split_audio.npy"))
            new_eye = np.load( os.path.join(sender_id_path, "x_split_eye.npy"), allow_pickle=True)
            new_text = np.load( os.path.join(sender_id_path, "x_split_text.npy"))
            new_hrv = np.load( os.path.join(sender_id_path, "x_split_hrv.npy"), allow_pickle=True)
            new_face = np.load( os.path.join(sender_id_path, "x_split_face.npy"), allow_pickle=True)
            new_label = np.load( os.path.join(sender_id_path, "y_label.npy"))
            
            for data in [new_audio, new_eye, new_text, new_hrv, new_face]:
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if isinstance(data[i][j], dict):
                            data[i][j] = 0
            
            count = len(new_audio)
            for i in range(count):
                audio.append( new_audio[i])
                eye.append( new_eye[i])
                text.append( new_text[i])
                hrv.append( new_hrv[i])
                face.append( new_face[i])
                label.append(new_label)
            
        # X = np.asarray(X).astype(np.float32)
        # 將所有串接好的陣列轉成 numpy 陣列
        data_audio = np.asarray(audio).astype(np.float32)
        data_eye = np.asarray(eye).astype(np.float32)
        data_text = np.asarray(text).astype(np.float32)
        data_hrv = np.asarray(hrv).astype(np.float32)
        data_face = np.asarray(face).astype(np.float32)
        fix_face = []
        for xs in data_face:
            temp = []
            for x in xs:
                temp.append(x)
        
            fix_face.append(np.array(temp))
        fix_face = np.asarray(fix_face).astype(np.float32)

        data_label = np.array(label)

        # Count unique values
        (unique, counts) = np.unique(data_label, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)

        y = to_categorical([int(val) for val in data_label], catagory)

        # print("audio shape:", data_audio.shape)
        # print("eye shape:", data_eye.shape)
        # print("text shape:", data_text.shape)
        # print("hrv shape:", data_hrv.shape)
        # print("face shape:", fix_face.shape)
        # print("label shape:", y.shape)
        
        return data_text, data_audio, data_hrv, data_eye, fix_face, y

def FiveModalFusion(EPOCH):
    """
    Feature-level fusion with concatenation of 5-modal features: text, audio, face, hrv, and eye-movement
    Classify 5 class: mild, moderate, severe, manic, normal
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 5
    TEST_SIZE = 0.25
    LABEL = ["Normal", "Mild", "Moderate", "Severe", "Manic"]
    ROOT = "/root/MatureDepressionAssessment/ru_test"
    # ROOT = "five_modal_five_class"
    
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"

    # Create directory for saving model and confusion matrix
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load data
    x_text, x_audio, x_hrv, x_eye, x_face, y = Dataset("/root/MatureDepressionAssessment/old_questionaire_teens", CATEGORY)
    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_hrv = np.nan_to_num(x_hrv)
    x_eye = np.nan_to_num(x_eye)
    x_face = np.nan_to_num(x_face)

    print("x_text shape:", x_text.shape)
    print("x_audio shape:", x_audio.shape)
    print("x_hrv shape:", x_hrv.shape)
    print("x_eye shape:", x_eye.shape)
    print("x_face shape:", x_face.shape)
    print("y shape:", y.shape)

    # Model
    model = five_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    losses, accuracies = [], []  # Lists to store loss and accuracy per epoch for plotting
    
    for train_index, test_index in rs.split(x_text):
        for e in range(1, EPOCH+1):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                hrv = np.expand_dims(x_hrv[index], axis=0)
                eye_movement = np.expand_dims(x_eye[index], axis=0)
                
                y_true = np.expand_dims(y[index], axis=0)
                
                
                model.fit(
                    x=[text, audio, face, hrv, eye_movement],
                    y=y_true,
                    #callbacks = tf.keras.callbacks.EarlyStopping(
                    #    monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='min',
                    #    baseline=None, restore_best_weights=True),
                    #batch_size = 2
                )

            # Evaluate
            if (e != 0) and (e%5 == 0):
                # init
                confusion_matrix = np.zeros((CATEGORY, CATEGORY))
                test_loss, test_accuracy = 0, 0
                for index in test_index:
                    text = np.expand_dims(x_text[index], axis=0)
                    audio = np.expand_dims(x_audio[index], axis=0)
                    face = np.expand_dims(x_face[index], axis=0)
                    hrv = np.expand_dims(x_hrv[index], axis=0)
                    eye_movement = np.expand_dims(x_eye[index], axis=0)

                    y_true = np.argmax(np.expand_dims(y[index], axis=0), axis=1)
                    # Calculate loss and accuracy
                    loss, accuracy = model.evaluate([text, audio, face, hrv, eye_movement], y_true, verbose=0)
                    test_loss += loss
                    test_accuracy += accuracy
                    y_pred = np.argmax(model.predict([text, audio, face, hrv, eye_movement]), axis=1)
                    confusion_matrix += tf.math.confusion_matrix(y_true, y_pred, num_classes=CATEGORY)

                # Average loss and accuracy for the test set
                test_loss /= len(test_index)
                test_accuracy /= len(test_index)
                
                # Append for plotting
                losses.append(test_loss)
                accuracies.append(test_accuracy)
                print(f"Epoch {e}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")


                # Plot confusion matrix
                confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
                print(confusion_matrix)

                # Confusion matrix with actual number of data
                cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                # plt.title("Confusion Matrix")
                # plt.ylabel("Ground Truth")
                # plt.xlabel("Predicted Value")
                img_path = f"{CM_PATH}/epoch{e}.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Confusion matrix with percentage of data
                uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
                cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d")
                plt.title(f"Confusion Matrix at Epoch {e}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig(f"{CM_PATH}/epoch{e}.png")
                plt.clf()

                # Plot and save confusion matrix with percentages
                cm_normalized = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]
                cm_df = pd.DataFrame(cm_normalized, index=LABEL, columns=LABEL)
                sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1)
                plt.title(f"Normalized Confusion Matrix at Epoch {e}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig(f"{CM_PATH}/epoch{e}_normalized.png")
                plt.clf()

                # Save model
                model_name = f"{MODEL_PATH}/epoch{e}.h5"
                model.save(model_name)
                print(f"{model_name} has been saved")
                
        # Plot loss and accuracy over epochs
        epochs_range = range(5, EPOCH + 1, 5)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, accuracies, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy over Epochs")
        plt.legend()
        plt.tight_layout()

        # Save plot without showing
        plot_path = f"{CM_PATH}/test_loss_accuracy.png"
        plt.savefig(plot_path)
        print(f"Loss and accuracy plot saved at {plot_path}")



def FourModalFusion(EPOCH):
    """
    Feature-level fusion with concatenation of 5-modal features: text, audio, face, hrv
    Classify 5 class: mild, moderate, severe, manic, normal
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 5
    TEST_SIZE = 0.25
    LABEL = ["Normal", "Mild", "Moderate", "Severe", "Manic"]
    ROOT = "/root/MatureDepressionAssessment/ru_test"
    # ROOT = "five_modal_five_class"
    
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"

    # Create directory for saving model and confusion matrix
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load data
    x_text, x_audio, x_hrv, x_eye, x_face, y = Dataset("/root/MatureDepressionAssessment/old_questionaire_teens", CATEGORY)
    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_hrv = np.nan_to_num(x_hrv)
    x_eye = np.nan_to_num(x_eye)
    x_face = np.nan_to_num(x_face)

    print("x_text shape:", x_text.shape)
    print("x_audio shape:", x_audio.shape)
    print("x_hrv shape:", x_hrv.shape)
    print("x_eye shape:", x_eye.shape)
    print("x_face shape:", x_face.shape)
    print("y shape:", y.shape)

    # Model
    model = four_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    losses, accuracies = [], []  # Lists to store loss and accuracy per epoch for plotting
    
    for train_index, test_index in rs.split(x_text):
        for e in range(1, EPOCH+1):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                hrv = np.expand_dims(x_hrv[index], axis=0)
                eye_movement = np.expand_dims(x_eye[index], axis=0)
                
                y_true = np.expand_dims(y[index], axis=0)
                
                
                model.fit(
                    x=[text, audio, face, hrv],
                    y=y_true,
                    #callbacks = tf.keras.callbacks.EarlyStopping(
                    #    monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='min',
                    #    baseline=None, restore_best_weights=True),
                    #batch_size = 2
                )

            # Evaluate
            if (e != 0) and (e%5 == 0):
                # init
                confusion_matrix = np.zeros((CATEGORY, CATEGORY))
                test_loss, test_accuracy = 0, 0
                for index in test_index:
                    text = np.expand_dims(x_text[index], axis=0)
                    audio = np.expand_dims(x_audio[index], axis=0)
                    face = np.expand_dims(x_face[index], axis=0)
                    hrv = np.expand_dims(x_hrv[index], axis=0)
                    eye_movement = np.expand_dims(x_eye[index], axis=0)

                    y_true = np.argmax(np.expand_dims(y[index], axis=0), axis=1)
                    # Calculate loss and accuracy
                    loss, accuracy = model.evaluate([text, audio, face, hrv], y_true, verbose=0)
                    test_loss += loss
                    test_accuracy += accuracy
                    y_pred = np.argmax(model.predict([text, audio, face, hrv]), axis=1)
                    confusion_matrix += tf.math.confusion_matrix(y_true, y_pred, num_classes=CATEGORY)

                # Average loss and accuracy for the test set
                test_loss /= len(test_index)
                test_accuracy /= len(test_index)
                
                # Append for plotting
                losses.append(test_loss)
                accuracies.append(test_accuracy)
                print(f"Epoch {e}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")


                # Plot confusion matrix
                confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
                print(confusion_matrix)

                # Confusion matrix with actual number of data
                cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                # plt.title("Confusion Matrix")
                # plt.ylabel("Ground Truth")
                # plt.xlabel("Predicted Value")
                img_path = f"{CM_PATH}/epoch{e}.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Confusion matrix with percentage of data
                uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
                cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d")
                plt.title(f"Confusion Matrix at Epoch {e}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig(f"{CM_PATH}/epoch{e}.png")
                plt.clf()

                # Plot and save confusion matrix with percentages
                cm_normalized = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]
                cm_df = pd.DataFrame(cm_normalized, index=LABEL, columns=LABEL)
                sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1)
                plt.title(f"Normalized Confusion Matrix at Epoch {e}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig(f"{CM_PATH}/epoch{e}_normalized.png")
                plt.clf()

                # Save model
                model_name = f"{MODEL_PATH}/epoch{e}.h5"
                model.save(model_name)
                print(f"{model_name} has been saved")
                
        # Plot loss and accuracy over epochs
        epochs_range = range(5, EPOCH + 1, 5)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, accuracies, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy over Epochs")
        plt.legend()
        plt.tight_layout()

        # Save plot without showing
        plot_path = f"{CM_PATH}/test_loss_accuracy.png"
        plt.savefig(plot_path)
        print(f"Loss and accuracy plot saved at {plot_path}")


def FiveModalFusion_Adults(EPOCH):
    """
    Feature-level fusion with concatenation of 5-modal features: text, audio, face, hrv, and eye-movement
    Classify 3 class: normal, mild, moderate
    """
    # Get current time
    nowTime = int(time.time())
    struct_time = time.localtime(nowTime)
    timestamp = time.strftime(f"%Y_%m_%d_%I_%M_%S", struct_time)
    
    # Parameters
    CATEGORY = 3
    TEST_SIZE = 0.25
    LABEL = ["Normal", "Mild", "Moderate"]
    ROOT = "five_modal_three_class"
    MODEL_PATH = f"{ROOT}/{timestamp}" # for saving model and confusion matrix
    CM_PATH = f"{MODEL_PATH}/confusion_matrix"

    # Create directory for saving model and confusion matrix
    for path in [ROOT, MODEL_PATH, CM_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load data
    x_text, x_audio, x_hrv, x_eye, x_face, y = Dataset("/root/MatureDepressionAssessment/new_questionaire_three_category_0614")
    # Replace Nan with 0
    x_text = np.nan_to_num(x_text)
    x_audio = np.nan_to_num(x_audio)
    x_hrv = np.nan_to_num(x_hrv)
    x_eye = np.nan_to_num(x_eye)
    x_face = np.nan_to_num(x_face)

    print("x_text shape:", x_text.shape)
    print("x_audio shape:", x_audio.shape)
    print("x_hrv shape:", x_hrv.shape)
    print("x_eye shape:", x_eye.shape)
    print("x_face shape:", x_face.shape)
    print("y shape:", y.shape)

    # Model
    model = five_fusion_model(CATEGORY)
    print(model.summary())

    rs = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=6)
    
    for train_index, test_index in rs.split(x_text):
        for e in range(1, EPOCH+1):
            # batch size = 1 since length of facial embedding is different
            for index in train_index:
                text = np.expand_dims(x_text[index], axis=0)
                audio = np.expand_dims(x_audio[index], axis=0)
                face = np.expand_dims(x_face[index], axis=0)
                hrv = np.expand_dims(x_hrv[index], axis=0)
                eye_movement = np.expand_dims(x_eye[index], axis=0)
                
                y_true = np.expand_dims(y[index], axis=0)
                
                
                model.fit(
                    x=[text, audio, face, hrv, eye_movement],
                    y=y_true,
                )

            # Evaluate
            if (e != 0) and (e%5 == 0):
                # init
                confusion_matrix = np.zeros((CATEGORY, CATEGORY))

                for index in test_index:
                    text = np.expand_dims(x_text[index], axis=0)
                    audio = np.expand_dims(x_audio[index], axis=0)
                    face = np.expand_dims(x_face[index], axis=0)
                    hrv = np.expand_dims(x_hrv[index], axis=0)
                    eye_movement = np.expand_dims(x_eye[index], axis=0)

                    y_true = np.argmax(np.expand_dims(y[index], axis=0), axis=1)
                    y_pred = np.argmax(model.predict([text, audio, face, hrv, eye_movement]), axis=1)
                    confusion_matrix += tf.math.confusion_matrix(y_true, y_pred, num_classes=CATEGORY)

                # Plot confusion matrix
                confusion_matrix = confusion_matrix.numpy() # tensor to numpy array
                print(confusion_matrix)

                # Confusion matrix with actual number of data
                cm_df = pd.DataFrame(confusion_matrix, index=LABEL, columns=LABEL)
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt="d") # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                # plt.title("Confusion Matrix")
                # plt.ylabel("Ground Truth")
                # plt.xlabel("Predicted Value")
                img_path = f"{CM_PATH}/epoch{e}.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Confusion matrix with percentage of data
                uniform_data = confusion_matrix/confusion_matrix.sum(axis=1)[:,None] # get percentage
                cm_df = pd.DataFrame(uniform_data, index=LABEL, columns=LABEL)
                plt.clf() # clear canvas
                plt.figure(figsize=(5, 5))
                ax = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1) # cmap: color style, annot: display quantity, fmt: format of quantity(default is scientific notation)
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                img_path = f"{CM_PATH}/epoch{e}_uniformed.png"
                plt.savefig(img_path)
                print(f"{img_path} has been saved")

                # Save model
                model_name = f"{MODEL_PATH}/epoch{e}.h5"
                model.save(model_name)
                print(f"{model_name} has been saved")

if __name__ == "__main__":
    start = datetime.now()

    # Dataset("/home/Transfer Learning Experiement/154_subject_dataset")
    # FiveModalFusion(EPOCH=100)
    # FourModalFusion(EPOCH=100)

    #FiveModalFusion_Adults(EPOCH=100)

    # three_modal_four_class(EPOCH=50)

    # five_modal_four_class(EPOCH=50)

    # three_modal_five_class(EPOCH=100)

    # four_modal_five_class(EPOCH=2)

    # five_modal_five_class(EPOCH=5)

    end = datetime.now()
    # print(f"執行時間：{end - start}")