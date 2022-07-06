# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 20:58:46 2022

@author: sdabadghao
"""
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Read latest file from datagrip 
df = pd.read_excel(r'C:\Users\sdabadghao\Dropbox (Personal)\personal\Learning python\shocks\shock_data.xlsx')


# Change header names by removing the '.' and replacing with '_'
df.columns = df.columns.str.replace('.', '_')

# # Replacing inf's and NaN's with 0
df.replace([np.inf, -np.inf, np.NaN], 0)


# Creating compound variables for features
collist1 = ['CCC', 'dio','dpo','dso']
collist2 =['DOL','WCAP','qckratio']
for columns in collist1:
    df[columns+'_delta'] = (df['shock_mean_'+columns]-df['preshock_mean_'+columns])/df['shock_mean_'+columns]

for columns in collist2:
    df[columns+'_delta'] = (df['shock_'+columns]-df['preshock_'+columns])/df['shock_'+columns]

# # Replacing inf's and NaN's with 0
df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)


# # Variables to keep
keepvars = ['CCC_delta','dio_delta','dpo_delta','dso_delta','DOL_delta',
                 'WCAP_delta','qckratio_delta','quantile_chge_DIO',
                 'quantile_chge_DSO','quantile_chge_DPO',
                 'quantile_chge_CCC','quantile_chge_WCAP','subgroups',
                 'average_perc_changes_10qrts_ltd','target']

# Set target variables
df['target'] = 0
df.loc[df['length'] < 11, 'target'] = 1

dft = df[keepvars]


# Validation and test sets
val_df = dft.sample(frac=0.2, random_state=1337)
train_df = dft.drop(val_df.index)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_df)
val_ds = dataframe_to_dataset(val_df)

# Batching
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


# Feature Pre-processing
# Subgroups is a categorical feature, encoded as an integer
# The remaining are continuous features which we will normalize

from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


# Categorical features encoded as integers
subgroups = keras.Input(shape=(1,), name="subgroups", dtype="int64")
subgroups_encoded = encode_categorical_feature(subgroups, "subgroups", train_ds, False)

# Numerical features
# 'CCC_delta','dio_delta','dpo_delta','dso_delta',DOL_delta','WCAP_delta','qckratio_delta','quantile_chge_DIO','quantile_chge_DSO','quantile_chge_DPO',
# 'quantile_chge_CCC','quantile_chge_WCAP','subgroups','average_perc_changes_10qrts_ltd','target'
CCC_delta = keras.Input(shape=(1,), name="CCC_delta")
dio_delta = keras.Input(shape=(1,), name="dio_delta")
dpo_delta = keras.Input(shape=(1,), name="dpo_delta")
dso_delta = keras.Input(shape=(1,), name="dso_delta")
DOL_delta = keras.Input(shape=(1,), name="DOL_delta")
WCAP_delta = keras.Input(shape=(1,), name="WCAP_delta")
qckratio_delta = keras.Input(shape=(1,), name="qckratio_delta")
# quantile_chge_DIO = keras.Input(shape=(1,), name="quantile_chge_DIO")
# quantile_chge_DPO = keras.Input(shape=(1,), name="quantile_chge_DPO")
# quantile_chge_DSO = keras.Input(shape=(1,), name="quantile_chge_DSO")
# quantile_chge_CCC = keras.Input(shape=(1,), name="quantile_chge_CCC")
# quantile_chge_WCAP = keras.Input(shape=(1,), name="quantile_chge_WCAP")
average_perc_changes_10qrts_ltd = keras.Input(shape=(1,), name="average_perc_changes_10qrts_ltd")

all_inputs = [subgroups,CCC_delta,dio_delta,dpo_delta,dso_delta,DOL_delta,
              WCAP_delta,qckratio_delta,
              #quantile_chge_DIO,quantile_chge_DSO,
              #quantile_chge_DPO,quantile_chge_CCC,quantile_chge_WCAP,
              average_perc_changes_10qrts_ltd]



CCC_delta_encoded = encode_numerical_feature(CCC_delta, "CCC_delta", train_ds)
dio_delta_encoded = encode_numerical_feature(dio_delta, "dio_delta", train_ds)
dpo_delta_encoded = encode_numerical_feature(dpo_delta, "dpo_delta", train_ds)
dso_delta_encoded = encode_numerical_feature(dso_delta, "dso_delta", train_ds)
DOL_delta_encoded = encode_numerical_feature(DOL_delta, "DOL_delta", train_ds)
WCAP_delta_encoded = encode_numerical_feature(WCAP_delta, "WCAP_delta", train_ds)
qckratio_delta_encoded = encode_numerical_feature(qckratio_delta, "qckratio_delta", train_ds)
# quantile_chge_DIO_encoded = encode_numerical_feature(quantile_chge_DIO, "quantile_chge_DIO_encoded", train_ds)
# quantile_chge_DPO_encoded = encode_numerical_feature(quantile_chge_DPO, "quantile_chge_DPO_encoded", train_ds)
# quantile_chge_DSO_encoded = encode_numerical_feature(quantile_chge_DSO, "quantile_chge_DSO_encoded", train_ds)
# quantile_chge_CCC_encoded = encode_numerical_feature(quantile_chge_CCC, "quantile_chge_CCC_encoded", train_ds)
# quantile_chge_WCAP_encoded = encode_numerical_feature(quantile_chge_WCAP, "quantile_chge_WCAP_encoded", train_ds)
average_perc_changes_10qrts_ltd_encoded = encode_numerical_feature(average_perc_changes_10qrts_ltd, "average_perc_changes_10qrts_ltd", train_ds)


all_features = layers.concatenate(
    [subgroups_encoded,
     CCC_delta_encoded,
     dio_delta_encoded,
     dpo_delta_encoded,
     dso_delta_encoded,
     DOL_delta_encoded,
     WCAP_delta_encoded,
     qckratio_delta_encoded,
     # quantile_chge_DIO_encoded,
     # quantile_chge_DSO_encoded,
     # quantile_chge_DPO_encoded,
     # quantile_chge_CCC_encoded,
     # quantile_chge_WCAP_encoded,
     average_perc_changes_10qrts_ltd_encoded
     ]
    )

x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# Fit the Model
model.fit(train_ds, epochs=50, validation_data=val_ds)
