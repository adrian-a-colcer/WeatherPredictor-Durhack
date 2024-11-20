import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

filename = "Durham-Observatory-daily-climatological-dataset.xlsx"
df = pd.read_excel(filename,sheet_name = "Durham Obsy - daily data 1843-")

dates = pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))
day_of_year = dates.dt.dayofyear

df = df.loc[df["YYYY"] >= 1950, 'Tmax °C':'Wet day']
df.dropna(inplace=True)

#forming periodicity in day values

df["Year sin"] = np.sin(day_of_year * (2 * np.pi / 365))
df["Year cos"] = np.cos(day_of_year * (2 * np.pi / 365))

#splitting the data

n = len(df)
train_df = df[0:int(0.7*n)]
valid_df = df[int(0.7*n):int(0.9*n)]
test_df = df[int(0.9*n):]
n_features = df.shape[1]

class GenWindow():
    def __init__(self, input_width, label_width, shift,
               train_df=train_df, valid_df=valid_df,
               test_df=test_df):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
 
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        self.window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.window_size)[self.input_slice]

        self.label_start = self.window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.window_size)[self.labels_slice]

    def split_window(self,features):
        inputs = features[:,self.input_slice]
        labels = features[:,self.labels_slice,:]

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices["Tmean °C"]
        max_n = min(3, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f"{"Tmean °C"} [normed]")
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label="Inputs", marker=".", zorder=-10)

            if plot_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, plot_col_index],
                        edgecolors="k", label="Labels", c="#2ca02c", s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, plot_col_index],
                        marker="X", edgecolors="k", label="Predictions",
                        c="#ff0e0e", s=64)

            if n == 0:
                plt.legend()

        plt.xlabel("Days")

    def make_ds(self,data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.window_size,
            sequence_stride = 1,
            shuffle = True,
            batch_size = 64
        )
        ds = ds.map(self.split_window)
        return ds
    @property
    def train(self):
        return self.make_ds(self.train_df)
    @property
    def valid(self):
        return self.make_ds(self.valid_df)
    @property
    def test(self):
        return self.make_ds(self.test_df)
    @property
    def example(self):
        result = getattr(self, "_example",None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result
    

OUTPUT_STEPS = 21
multistep_window = GenWindow(input_width=21,label_width=OUTPUT_STEPS,shift=OUTPUT_STEPS)

multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(252, return_sequences=False),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(252),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate = 0.3),
    tf.keras.layers.Dense(OUTPUT_STEPS*n_features,
                          kernel_initializer=tf.initializers.zeros()),

    tf.keras.layers.Reshape([OUTPUT_STEPS, n_features])
])

#TRAINING
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
                                                  patience = 10,
                                                  verbose = 1)

callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.1,
                                                min_lr=1e-3,
                                                patience=7,
                                                verbose=1)

multi_lstm_model.compile(loss = tf.losses.MeanSquaredError(),
                       optimizer = tf.keras.optimizers.RMSprop(),
                       metrics = [tf.keras.metrics.MeanAbsoluteError()])

history = multi_lstm_model.fit(multistep_window.train, epochs = 50,
                    validation_data = multistep_window.valid,
                    callbacks = [callback_reduce_lr,early_stopping])

multistep_window.plot(multi_lstm_model)
plt.show()
