#  -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import time

from tensorflow import keras
from keras.models import Model
from Position_Embedding import Position_Embedding
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D,Conv1D,add,Flatten,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras import Input
from attenTion import MultiHeadAttention
from sklearn.metrics import confusion_matrix
from keras import backend as K
from datetime import datetime
# from plotting_confusion_matrix import plot_confusion_matrix
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


current_file_path=os.path.dirname(os.path.abspath(__file__))
now = datetime.now()
data_path = current_file_path+'/Five_land_cover_HRRP_data/' 
batch_size = 32 #64
nb_epoch = 100
nb_features = 145 # The length of HRRP
embedding_dim = 32

def parameters_arg():
    parser = argparse.ArgumentParser(description='parameters related to training details')
    # type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument('--file_name', type=str, default='HRRP.csv',help='file name')
    args = parser.parse_args()
    return args


def encode(data):
    label_encoder = LabelEncoder().fit(data.scene)
    labels = label_encoder.transform(data.scene) # 将种类转换为某个整数来表示，这里是一个整数型数组，无论是否重复都会出现相应的int数
    classes = list(label_encoder.classes_) # 统计scene有几种，构造字典，这里是字符型，重复的只出现一次

    train_1 = data.drop(['scene'], axis=1) # 训练集是去掉前一列
    return train_1, labels, classes,label_encoder

def stanDard(data):
    scaler = StandardScaler().fit(data.values)
    scaled_data = scaler.transform(data.values)
    return scaled_data

def attention():    
    print('Model building ... ')
    inputs = Input(shape=(nb_features,1), name="inputs") # 张量
    embedding = Conv1D(embedding_dim, 1, activation='relu', padding='same')(inputs)# 1
    embedding = Position_Embedding()(embedding)  # 加效果更好
    x1 = MultiHeadAttention(2, 16)([embedding,embedding,embedding])# 2
    x2 = add([x1,embedding]) # 4
    x2 = keras.activations.selu(x2) # 可有可无,影响不大,不过似乎收敛更快一些
    x3 = MultiHeadAttention(2, 16)([x2,x2,x2])
    x4 = add([x2,x3])
    x4 = keras.activations.selu(x4) # 可有可无,影响不大,不过似乎收敛更快一些
    x5 = MultiHeadAttention(2, 16)([x4,x4,x4])
    x6 = add([x4,x5])
    x7 =Conv1D(1, 1, activation='tanh', padding='same')(x6)# 6
    x8 = Flatten()(x7)# 7
    outputs = Dense(num_classes, activation='softmax')(x8)
    model = Model(inputs = inputs, outputs=outputs)
    print(model.summary())
    return model

def attention_train(X_train_r,X_test_r,y_test,y_train):    
    model = attention()
    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.9, epsilon=1e-8), 
        loss='categorical_crossentropy', metrics=['accuracy'])#beta_1=0.9, beta_2=0.9
    print("Model Training ... ")
    es = EarlyStopping(patience=30) # 连续30个epoch出现负增长停止训练
    model.fit(X_train_r, y_train, 
        batch_size=batch_size, epochs=nb_epoch, validation_split=0.3, callbacks=[es])
    model.save(current_file_path+'/models/'+now.strftime("%H_%M_%S")+'.h5')
    
    #  model.fit函数还可以继续划分训练集和验证集，基于split的比例
    test_metrics = model.evaluate(X_test_r, y_test, batch_size=batch_size, verbose=0)
    print("loss on Test: %.4f" % test_metrics[0])
    print("accu on Test: %.4f" % test_metrics[1])
    print("Evaluating model...")
    start_time = time.time()
    predictions = model.predict(X_test_r)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Time taken for inference on one sample: {inference_time / len(X_test_r):.6f} seconds")
    
    predictions=np.argmax(predictions,axis=1)
    y_test = np.argmax(y_test,axis=1)
    fig2 = plt.figure(figsize=(16, 9), dpi=800)
    cm = confusion_matrix(y_true=y_test, y_pred=predictions)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Water', 'Grass', 'Soil'],
                yticklabels=['Water', 'Grass', 'Soil'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    fig2.savefig('confusion_matrix_result/confuse_matrix.png')
    print(cm)
    # plot_confusion_matrix(cm, classes, "test Confusion Matrix")  # 绘制混淆矩阵


def visulization_featureMap(model):
    #keras输出中间某一层
    model = keras.models.load_model(r'resnet1d.h5')
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[10].output])
    layer_output = get_3rd_layer_output([X_train_r])[0]



if __name__ == '__main__':
    opt = parameters_arg()
    # encoding training samples
    HRRP_data = pd.read_csv(data_path+opt.file_name) 
    HRRPdata, labels, classes, label_encoder = encode(HRRP_data)
    # standardize training samples
    scaled_data =  stanDard(HRRPdata)
    num_classes = len(classes)

    # split train data into train and validation9 
    sss = StratifiedShuffleSplit(test_size=0.2,random_state = 42)
    for train_index, test_index in sss.split(scaled_data, labels):
        X_train, X_test = scaled_data[train_index], scaled_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
    # add a dimention    
    X_train_r = np.zeros((len(X_train), nb_features, 1))#只有一个特征，所以第三个维度是1，全0
    X_train_r[:, :, 0] = X_train[:, :nb_features]
    X_test_r = np.zeros((len(X_test), nb_features, 1))#只有一个特征，所以第三个维度是1，全0
    X_test_r[:, :, 0] = X_test[:, :nb_features]
    traindm = np.zeros((len(HRRPdata), nb_features, 1))#只有一个特征，所以第三个维度是1，全0
    traindm[:, :, 0] = scaled_data[:, :nb_features]
    # encoding for labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    attention_train(X_train_r,X_test_r,y_test,y_train)    





