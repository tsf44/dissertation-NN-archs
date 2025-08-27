import numpy as np
import os
import matplotlib.pyplot as plt
import time
from scipy.io import savemat, loadmat
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Lambda, \ 
                                    MaxPooling1D, Flatten, \ 
                                    LeakyReLU, Concatenate, \
                                    BatchNormalization


class VanillaSHM(Model):
def __init__(self, input_shape, latent_shape, output_shape,
             pool_scaler=None, **kwargs):
    super().__init__(**kwargs)
    self.latent_dim = latent_shape[1] - 3
    self.output_dim = output_shape[1]
    self.pool_scaler = pool_scaler
    self.model = self.get_model(input_shape)
    self.total_loss = tf.keras.metrics.Mean(name='total_loss')

@property
def metrics(self):
    return [self.total_loss]

def get_model(self, input_shape):
    # LeakyReLU Hyperparameter
    alpha = 0.01  # -> To match Lin, et al (2017)
    # Scaling values for pool_size in MaxPooling1D()
    if self.pool_scaler is None:
        ps = [1, 1, 1]
    else:
        ps = self.pool_scaler

    # --- Input Layer ---
    x_input = tf.keras.Input(shape=input_shape[1:])
    # Add fft layer and split into real and imaginary parts
    x = Lambda(lambda v: tf.signal.fft(
        tf.cast(v, tf.complex64))
        )(x_input)
    x = Concatenate()([tf.math.real(x), tf.math.imag(x)])
    x = BatchNormalization()(x)
    # --- 1st Layer ---
    # 1a)
    h = Conv1D(filters=32, kernel_size=16, padding='same')(x)
    h = LeakyReLU(alpha=alpha)(h)
    # 1b)
    h = Conv1D(filters=32, kernel_size=16, padding='same')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=alpha)(h)
    # 1c)
    h = MaxPooling1D(pool_size=4 * ps[0])(h)
    # --- 2nd Layer ---
    # 2a)
    h = Conv1D(filters=32, kernel_size=16, padding='same')(h)
    h = LeakyReLU(alpha=alpha)(h)
    # 2b)
    h = Conv1D(filters=32, kernel_size=16, padding='same')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=alpha)(h)
    # 2c)
    h = MaxPooling1D(pool_size=4 * ps[1])(h)
    # --- 3rd Layer ---
    # 3a)
    h = Conv1D(filters=128, kernel_size=16, padding='same')(h)
    h = LeakyReLU(alpha=alpha)(h)
    # 3b)
    h = Conv1D(filters=128, kernel_size=16, padding='same')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=alpha)(h)
    # 3c)
    h = MaxPooling1D(pool_size=4 * ps[2])(h)
    # --- 4th Layer ---
    h = Flatten()(h)
    # --- 5th Layer ---
    z_phi = Dense(self.latent_dim, name='z_phi')(h)
    z_freqs = Dense(3, name='z_freqs')(h)
    z = Concatenate()([z_phi, z_freqs])
    # --- 6th Layer ---
    h = Dense(256)(z)
    h = LeakyReLU(alpha=alpha)(h)
    # --- 7th Layer ---
    h = Dense(128)(h)
    h = LeakyReLU(alpha=alpha)(h)
    # --- Output Layer ---
    health = Dense(self.output_dim)(h)
    return Model(inputs=x_input,
                 outputs=health,
                 name='vanilla_classifier')

def fit(self, X_train, y_train, epochs=1, batch_size=10,
        verbose=1):
    # Convert all data into single dataset as Dataset dtype
    dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
        )
    # Set batch size of dataset
    dataset = dataset.batch(batch_size)

    # Instantiate history dictionary
    history = {m.name: [] for m in self.metrics}

    # Perform training by epoch
    for epoch in range(epochs):
        print("Epoch: {:1d}/{:1d}".format(epoch + 1, epochs))
        start_time = time.time()
        # Perform training for each batch of data per epoch
        # Returns a single batch per loop iteration
        for x, y in dataset:
            train_metrics = self.train_step(x, y)
        end_time = time.time()
        T = end_time - start_time
        if (verbose == 1) or (verbose == 2):
            logs = ""
            for m in self.metrics[:4]:
                if train_metrics[m.name] < 0.0001:
                    logs = logs \
                        + "{:}: {:.4E} - ".format(
                            m.name, train_metrics[m.name]
                            )
                else:
                    logs = logs \
                        + "{:}: {:.4f} - ".format(
                            m.name, train_metrics[m.name]
                            )
            strA = "{:1d}/{:1d} - {:1g}s - ".format(
                    len(dataset), len(dataset), round(T)
                    )
            strZ = "{:1g}s/epoch - {:3g}ms/step".format(
                    round(T), round(1000*T/len(dataset))
                    )
            print(strA, logs, strZ, sep="")
        # Populate history
        for m in self.metrics:
            history[m.name].append(m.result().numpy())
            # Reset metric values
            m.reset_states()
    return history

def train_step(self, X, y):
    # Define loss objects
    loss_object = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        # Predict structural health from latent space
        y_pred = self.model(X)
        # --- Calculate losses ---
        # Data Loss
        total_loss = loss_object(y, y_pred)
    # Get gradients and apply to optimizer
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(
        zip(grads, self.trainable_weights)
        )
    # Update metrics
    self.total_loss.update_state(total_loss)
    return {m.name: m.result().numpy() for m in self.metrics}

def predict(self, X):
    y_pred = self.model.predict(X)
    return y_pred

def freeze_layers(self):
    modelA_layers = 26
    for ii in range(modelA_layers):
        self.model.layers[ii].trainable = False
    return

def unfreeze_layers(self):
    modelA_layers = 26
    for ii in range(modelA_layers):
        self.model.layers[ii].trainable = True
    return

def score(self, y_true, y_pred, loc=[3, 5, 10]):
    # Covariance
    covar = self.local_covariance(y_true, y_pred)
    # Correlation
    R = self.local_correlation(y_true, y_pred)
    # Coefficient of Determination
    r2_score = self.local_r2_score(y_true, y_pred)
    # Error metrics
    [mae, mse, std_error] = self.local_error_metrics(y_true,
                                                     y_pred)

    # Combine metrics into dictionary
    metrics = {'Cov': covar, 'R': R, 'r2_score': r2_score,
               'MAE': mae, 'MSE': mse, 'std_error': std_error}

    # Strings to print
    str1 = 'Cov(y_true, y_pred) = ['
    str2 = 'R(y_true, y_pred) = ['
    str3 = 'R^2(y_true, y_pred) = ['
    str4 = 'MAE(y_true, y_pred) = ['
    str5 = 'MSE(y_true, y_pred) = ['
    str6 = 'std(y_true - y_pred) = ['
    for ii in range(len(loc)):
        idx = loc[ii]
        str1 += '{:.4g}'.format(covar[idx][0, 1])
        str2 += '{:.4g}'.format(R[idx][0, 1])
        str3 += '{:.4g}'.format(r2_score[idx])
        str4 += '{:.4g}'.format(mae[idx])
        str5 += '{:.4g}'.format(mse[idx])
        str6 += '{:.4g}'.format(std_error[idx])
        if ii == len(loc) - 1:
            str1 += ']'
            str2 += ']'
            str3 += ']'
            str4 += ']'
            str5 += ']'
            str6 += ']'
        else:
            str1 += ', '
            str2 += ', '
            str3 += ', '
            str4 += ', '
            str5 += ', '
            str6 += ', '

    print('Vanilla Model')
    print(str1)
    print(str2)
    print(str3)
    print(str4)
    print(str5)
    print(str6)
    return metrics

def local_covariance(self, y_true, y_pred):
    # Determine the number of positions
    npos = y_true.shape[1]
    # For each position
    covariance = [np.cov(y_true[:, ii], y_pred[:, ii]
                         ) for ii in range(npos)]
    return covariance

def local_correlation(self, y_true, y_pred):
    # Determine the number of positions
    npos = y_true.shape[1]
    # For each position
    correlation = [np.corrcoef(y_true[:, ii], y_pred[:, ii]
                               ) for ii in range(npos)]
    return correlation

def local_r2_score(self, y_true, y_pred, mu=1e-12):
    # Ensure no division by zero
    y_true = y_true + mu
    # Coefficient of determination
    #     see sklearn documentation as r2_score = (1 - u/v)
    u = ((y_true - y_pred) ** 2).sum(axis=0)
    v = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2_score = (1 - u/v)
    return r2_score

def local_error_metrics(self, y_true, y_pred):
    # Define error between expected and predicted
    error = y_true - y_pred
    # Mean absolute error by output location
    mae = np.mean(np.abs(error), axis=0)
    # Mean square error by output location
    mse = np.mean(error ** 2, axis=0)
    # Standard deviation of error by output location
    std_of_error = np.std(error, axis=0)
    return [mae, mse, std_of_error]

def save_model(self, modelName, history=None, 
               save_path=os.getcwd()):
    # --- Save history data as .mat file ---
    if type(history) is dict:
        savemat(os.path.join(
            save_path,
            "History_" + modelName + ".mat"), history
            )
    else:
        print('---No history data saved for this model---')
    # --- Save weights of model ---
    self.model.save_weights(
        os.path.join(
            save_path,
            "Weights_" + modelName + "_Model.h5"
            )
        )

def load_trained_weights(self, file):
    # --- Load weights .h5 file ---
    self.model.load_weights(file)

def load_history(self, path, save_key):
    history = loadmat(
        os.path.join(
            path,
            ('History_VanillaSHM_' + save_key + '.mat')
            )
        )
    for k in history.keys():
        if type(history[k]) is np.ndarray:
            history[k] = np.squeeze(history[k])
    return history

