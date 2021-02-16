#%%
import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
# import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
# from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
print("Done")
#%%
df = pd.read_csv('data.csv')
# %%
df.head()
df.shape
# %%
df.isnull().sum()
# %%  Reshaping the Data
df["Image"]=df["Image"].apply(lambda x : np.fromstring(x,dtype=int,sep=" ").reshape(96,96))
df.head()
# %%    Visualizing
i=np.random.randint(1,len(df))
plt.imshow(df["Image"][i],cmap='gray')
for j in range(1,31,2):
    plt.plot(df.loc[i][j-1],df.loc[i][j],"rx")
# %%
df.isnull().sum()
# %% 
fiq=plt.figure(figsize=(20,20))
for i in range(16):
    ax=fiq.add_subplot(4,4,i+1)
    plt.imshow(df["Image"][i],cmap='gray')
    for j in range(1,31,2):
        plt.plot(df.loc[i][j-1],df.loc[i][j],"rx")
# %%               Image Augumentation
df_copy=df.copy()
df_copy.head()
# %%                   Flipping the images by 180 degrees
df_copy["Image"]=df_copy["Image"].apply(lambda x :np.flip(x,axis=0))
columns=df_copy.columns
for i in range(len(columns)):
    if i%2 != 0:
        df_copy[columns[i]]=df_copy[columns[i]].apply(lambda x : 96. - float(x))

# %%                     Flipping the images by 90 degrees

df_copy["Image"]=df_copy["Image"].apply(lambda x :np.flip(x,axis=1))
columns=df_copy.columns
for i in range(len(columns)-1):
    if i%2 == 0:
        df_copy[columns[i]]=df_copy[columns[i]].apply(lambda x : 96. - float(x))

# %%
aug_df=np.concatenate((df,df_copy))
aug_df.shape
# %%        Increasing the brigthness of the images  
df_copy2=df.copy()
# %%
import random
df_copy2["Image"]=df_copy2["Image"].apply(lambda x : np.clip(random.uniform(1.5,2)*x,0.0,255.0))
# %%
fig=plt.figure(figsize=(20,20))
for i in range(16):
    ax=fig.add_subplot(4,4,i+1)
    plt.imshow(df_copy2["Image"][i],cmap="gray")
# %%
fig=plt.figure(figsize=(20,20))
for i in range(16):
    ax=fig.add_subplot(4,4,i+1)
    plt.imshow(df["Image"][i],cmap="gray")
# %%
aug_df=np.concatenate((aug_df,df_copy2))
aug_df.shape
# %%   Data Normalizationn
img=aug_df[-1]
img=img/255.0

# %%

# %%

# %%

# %%
