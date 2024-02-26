from everywhereml.sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from everywhereml.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from everywhereml.preprocessing import Pipeline, MinMaxScaler, Window, SpectralFeatures
print(pd. __version__)
x_sine = np.loadtxt("2_sine.txt",
                 delimiter=",", dtype=np.float16, encoding="utf8")
x_square = np.loadtxt("2_sq.txt",
                 delimiter=",", dtype=np.float16, encoding="utf8")


s_pipeline = Pipeline(name='SignalPipeline', steps=[
    MinMaxScaler(),
    # shift can be an integer (number of samples) or a float (percent)
    Window(length=128, shift=0.3),
    # order can either be 1 (first-order features) or 2 (add second-order features)
    SpectralFeatures(order=2)
])

size_sine =  x_sine.shape
length_sin = size_sine[0]
size_sq =  x_square.shape
length_sq = size_sq[0]

sin_labels = np.full((length_sin, 1), 0, np.float32)
sq_labels = np.full((length_sq, 1), 1, np.float32)




combined_x = np.concatenate((x_sine, x_square), axis=0)
combined_y = np.concatenate((sin_labels, sq_labels), axis=0)

size_c =  combined_x.shape
print("Combined shape")

length_c = size_c[0]
combined_x = np.nan_to_num(combined_x)
all = np.empty([length_c, 5])
print(all.shape)
for i in range(length_c):
       
        min = np.nanmin(combined_x[i,:])
        subtracted = combined_x[i,:] - min
        subtracted = np.nan_to_num(subtracted)
        max = np.nanmax(subtracted)
        divided = subtracted / max
        combined_x[i,:] = divided
        all[i,0] = np.nanmean(combined_x[i,:])
        all[i,1] = np.nanvar(combined_x[i,:])
        all[i,2] = np.nanstd(combined_x[i,:])
        all[i,3] = np.nansum(combined_x[i,:])
        squared = np.square(combined_x[i,:])
        all[i,4] = np.nansum(squared) / 255

all = np.nan_to_num(all[:,:])
column_values = ['mean','var','std','sum','sq']
combined_y = combined_y.flatten()
combined_y = combined_y.tolist()
columns_n = np.arange(256)
print(combined_y)

df = pd.DataFrame(data = all, 
                  index = combined_y
                  )
print(df.describe())

signal_classifier = RandomForestClassifier(n_estimators=20, max_depth=20)
train = df.sample(frac=0.69,random_state=200)
train.reset_index()

test = df.sample(frac=0.3,random_state=200)
test.reset_index()

signal_classifier.fit(train)
print('Score on test set: %.2f' % signal_classifier.score(test))

print(signal_classifier.to_arduino_file(
    'Classifier.h', 
    instance_name='forest', 
    class_map=df.class_map
))