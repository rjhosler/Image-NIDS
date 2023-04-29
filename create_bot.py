import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import bottleneck as bn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import hashlib
import os
pd.set_option('display.float_format', lambda x: '%.1f' % x)
pd.set_option('display.max_columns', None)
SIZE = 32
ADDITION = ""

RED_CHANNEL = np.load("Channel-Vectors/RED_CHANNEL.npy")
GREEN_CHANNEL = np.load("Channel-Vectors/GREEN_CHANNEL.npy")
BLUE_CHANNEL = np.load("Channel-Vectors/BLUE_CHANNEL.npy")

def df_to_features(file, mode):
    df = pd.read_csv(file)
    df = df.dropna()
    col_names = df.columns.values.tolist()
    exclude_names = ['pkSeqID', 'proto', 'saddr', 'sport', 'daddr', 
                 'dport', 'attack', 'category', 'subcategory']
    keys = []
    for item in col_names:
        if item not in exclude_names:
            keys.append(item)

    features = df[keys]
    features = (features - features.min()) / (features.max() - features.min())
    features = features.fillna(0)
    features = features.values
    
    labels = df['category'].values
    label_list = np.unique(labels)
      
    for j in range(len(label_list)):
        DIR = "BOT-IOT/Images/"+mode+"/"+label_list[j]
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        for i in range(len(features)):
            if labels[i] == label_list[j]: 
                red = np.convolve(RED_CHANNEL, features[i], mode='same')
                green = np.convolve(GREEN_CHANNEL, features[i], mode='same')
                blue = np.convolve(BLUE_CHANNEL, features[i], mode='same')

                # convert values to 0 - 255 int8 format
                red = (red * 255 / np.max(red)).astype('uint8')
                green = (green * 255 / np.max(green)).astype('uint8')
                blue = (blue * 255 / np.max(blue)).astype('uint8')

                img = np.zeros((SIZE,SIZE,3))
                ch1 = red.reshape((SIZE,SIZE))
                ch2 = green.reshape((SIZE,SIZE))
                ch3 = blue.reshape((SIZE,SIZE))

                img[:,:,0] = ch1
                img[:,:,1] = ch2
                img[:,:,2] = ch3
                img = img.astype(np.uint8)

                im = Image.fromarray(img)
                im = im.resize((SIZE, SIZE), Image.ANTIALIAS)
                im.save(DIR+"/"+str(i)+".png","PNG")
                
if __name__ == "__main__":
    df_to_features("BOT-IOT/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv", 'Train')
    df_to_features("BOT-IOT/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv", 'Test')