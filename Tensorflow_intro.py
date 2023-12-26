import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers


#load Data

df_data = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

df_train = df_data.sample(frac=0.8,random_state=0)
df_test = df_data.drop(df_train.index)

print(df_train.head())

train_labels = df_train.pop("Age")
test_labels = df_test.pop("Age")

#Create the model

regression_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])
regression_model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),optimizer=tf.keras.optimizers.Adam())
#Train the model
regression_model.fit(df_train,train_labels,epochs=40,batch_size=100)

#evaluation of the model

eval = regression_model.evaluate(df_test,test_labels)

#Using the model to predict

abalone = np.array([[ 0.515, 0.405, 0.14, 0.7665, 0.3075, 0.1495, 0.2605]]) #Abalone data to predict

age = regression_model.predict(abalone)

pr_age = round(age[0][0],3)
pr_mape = round(eval,3)


print("The estimated age of the specified abalone is "+ str(pr_age) + " +/- " + str(pr_mape) +"%")


