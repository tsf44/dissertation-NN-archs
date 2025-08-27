import numpy as np
from scipy.io import savemat
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Flatten, Reshape, \
                                    Lambda, Concatenate
                                    
import os
import time


class Physics_EqCircuit(Model):
    def __init__(self, input_shape, latent_shape, seed=None,
                 task=None, Zmag=False, **kwargs):
        super().__init__(**kwargs)
        # Input/Output Dimensions
        self.input_dim = input_shape[1:]
        self.latent_dim = latent_shape[1]
        # Seed for random number generator
        self.seed = seed
        # Determine if input is the impedance magnitude or complex 
        self.Zmag = Zmag  # <bool>
        # Get the task-specific encoder
        if task:
            self.encoder = self.get_MultiTask_encoder(input_shape, Zmag)
        else:
            self.encoder = self.get_DNN_encoder(input_shape)
        # Define loss metrics
        self.data_loss = tf.keras.metrics.Mean(name='data_loss')
        self.physics_loss = tf.keras.metrics.Mean(name='physics_loss')
        self.Re_loss = tf.keras.metrics.Mean(name='Re_loss')
        self.Rms_loss = tf.keras.metrics.Mean(name='Rms_loss')
        self.Cms_loss = tf.keras.metrics.Mean(name='Cms_loss')
        self.Mms_loss = tf.keras.metrics.Mean(name='Mms_loss')
        self.BLsq_loss = tf.keras.metrics.Mean(name='BLsq_loss')
        self.Le_loss = tf.keras.metrics.Mean(name='Le_loss')
        self.Zr_loss = tf.keras.metrics.Mean(name='Zr_loss')
        self.Zi_loss = tf.keras.metrics.Mean(name='Zi_loss')
        self.total_loss = tf.keras.metrics.Mean(name='total_loss')

    @property
    def metrics(self):
        return [self.total_loss, self.data_loss, self.physics_loss,
                self.Rms_loss, self.Cms_loss, self.Mms_loss, self.BLsq_loss,
                self.Re_loss, self.Le_loss, self.Zr_loss, self.Zi_loss]
        
     def get_DNN_encoder(self, input_shape):
        # LeakyReLU Hyperparameter
        alpha = 0.2
        # Dropout Rate
        rate = 0.03
        # Dense layer connections per branch
        units = [3*25, 3*15, 3*10, 3]

        # Define input layer
        z_input = tf.keras.Input(shape=(input_shape[1],input_shape[2]),
                                 name='Z_input')

        
        # Slice inputs into real and imaginary parts
        z_real = Lambda(lambda z: tf.slice(z, [0, 0, 0], [-1, input_shape[1]-2, 1]),
                        name='Real_Z')(z_input)
        z_imag = Lambda(lambda z: tf.slice(z, [0, 0, 1], [-1, input_shape[1]-2, 1]),
                        name='Imag_Z')(z_input)

        # Real Branch
        hr = Flatten()(z_real)
        # Layer 1
        hr = Dense(units[0], name='xr_Dense_1')(hr)
        hr = LeakyReLU(alpha=alpha)(hr)
        hr = Dropout(rate=rate, seed=self.seed, name='xr_Dropout_1')(hr)
        # Layer 2
        hr = Dense(units[1], name='xr_Dense_2')(hr)
        hr = LeakyReLU(alpha=alpha)(hr)
        hr = Dropout(rate=rate, seed=self.seed, name='xr_Dropout_2')(hr)
        # Layer 3
        hr = Dense(units[2], name='xr_Dense_3')(hr)
        hr = LeakyReLU(alpha=alpha)(hr)
        hr = Dropout(rate=rate, seed=self.seed, name='xr_Dropout_3')(hr)
        # Branch Output
        hr = Dense(units[3], name='Re_x_hat')(hr)
        hr = LeakyReLU(alpha=alpha)(hr)

        # Imaginary Branch
        hi = Flatten()(z_imag)
        # Layer 1
        hi = Dense(units[0], name='xi_Dense_1')(hi)
        hi = LeakyReLU(alpha=alpha)(hi)
        hi = Dropout(rate=rate, seed=self.seed, name='xi_Dropout_1')(hi)
        # Layer 2
        hi = Dense(units[1], name='xi_Dense_2')(hi)
        hi = LeakyReLU(alpha=alpha)(hi)
        hi = Dropout(rate=rate, seed=self.seed, name='xi_Dropout_2')(hi)
        # Layer 3
        hi = Dense(units[2], name='xi_Dense_3')(hi)
        hi = LeakyReLU(alpha=alpha)(hi)
        hi = Dropout(rate=rate, seed=self.seed, name='xi_Dropout_3')(hi)
        # Branch Output
        hi = Dense(units[3], name='Im_x_hat')(hi)
        hi = LeakyReLU(alpha=alpha)(hi)

        # Concatenate branch outputs as encoder output layer
        x = Concatenate(name='x_all')([hr, hi])
        return Model(inputs=z_input, outputs=x, name='encoder')

    def get_MultiTask_encoder(self, input_shape, Zmag):
        # LeakyReLU Hyperparameter
        alpha = 0.2

        # Dropout Rate
        rate = 0.03

        # Input layer
        z_input = tf.keras.Input(shape=(input_shape[1], input_shape[2]),
                                 name='Z_input')

        if Zmag is True:
            # Slice inputs into real and imaginary parts
            z_real = Lambda(lambda z: tf.slice(z, [0, 0, 0],
                                               [-1, input_shape[1]-2, 1]),
                            name='Real_Z')(z_input)
            # Make inputs to real/imag branches identical
            hr = Flatten(name='Flat_z_real')(z_real)
            hi = Flatten(name='Flat_z_imag')(z_real)
        else:
            # Slice inputs into real and imaginary parts
            z_real = Lambda(lambda z: tf.slice(z, [0, 0, 0],
                                               [-1, input_shape[1]-2, 1]),
                            name='Real_Z')(z_input)
            z_imag = Lambda(lambda z: tf.slice(z, [0, 0, 1],
                                               [-1, input_shape[1]-2, 1]),
                            name='Imag_Z')(z_input)
            # Define real/imag branch inputs
            hr = Flatten(name='Flat_z_real')(z_real)
            hi = Flatten(name='Flat_z_imag')(z_imag)

        # --- Instantiate loop variables ---
        # Number of individual connections per TS parameter
        units = [25, 15, 10, 1]
        # Hidden layer output list
        h_out = []
        x_out = []
        for branch in ['Real', 'Imag']:
            # Define parameters
            if branch == 'Real':
                x_name = ['Rms', 'BLsq', 'Re']
            else:
                x_name = ['Mms', 'Cms', 'Le']

            for idx, name in enumerate(x_name):
                h_ = []
                for layer, unit in enumerate(units):
                    # Define current layer
                    lay = str(layer + 1)
                    
                    # Set hidden layer input
                    if layer == 0:
                        if branch == 'Imag':
                            h_input = hi
                        else:
                            h_input = hr
                    else:
                        h_input = h_out
                        
                    # Define hidden layers
                    if layer != len(units) - 1:
                        h_ = Dense(unit, name=name + '_Dense_' + lay)(h_input)
                        h_ = LeakyReLU(alpha=alpha,
                                       name=name + '_LeakyReLU_' + lay)(h_)
                        h_out = Dropout(rate=rate, seed=self.seed,
                                        name=name + '_Dropout_' + lay)(h_)
                    else:
                        # Output Layer
                        x_ = Dense(unit, name=name + '_x_pred')(h_out)
                        x_pred = LeakyReLU(name=name + '_pred')(x_)
                        # Append output x_pred to x_out
                        x_out.append(x_pred)
        # Concatenate branch outputs as encoder output layer
        x = Concatenate(name='x_all')(x_out)
        return Model(inputs=z_input, outputs=x, name='encoder')

    def calc_Z(self, w, Re, Le, Cms, Mms, Rms, BLsq):
        if w.shape == Re.shape:
            Zr = Re + \
                tf.divide(
                    tf.multiply(
                        tf.multiply(Rms,
                                    tf.multiply(BLsq, tf.square(Cms))),
                        tf.square(w)
                        ),
                    (tf.square(
                            1 - tf.multiply(
                                    tf.multiply(Mms, Cms),
                                    tf.square(w))
                            )
                        + tf.square(
                            tf.multiply(
                                tf.multiply(Rms, Cms),
                                w)
                            ))
                    )
            Zi = tf.multiply(Le, w) + \
                tf.divide(
                    tf.multiply(
                        tf.multiply(
                            tf.multiply(BLsq, Cms),
                            w),
                        1 - tf.multiply(
                                 tf.multiply(Mms, Cms),
                                 tf.square(w)
                                 )
                        ),
                    (tf.square(
                            1 - tf.multiply(
                                    tf.multiply(Mms, Cms),
                                    tf.square(w))
                            )
                        + tf.square(
                            tf.multiply(
                                tf.multiply(Rms, Cms),
                                w)
                            ))
                    )
        else:
            Zr = Re + \
                tf.divide(
                    tf.matmul(
                        tf.multiply(Rms,
                                    tf.multiply(BLsq, tf.square(Cms))),
                        tf.square(w)
                        ),
                    (tf.square(
                            1 - tf.matmul(
                                    tf.multiply(Mms, Cms),
                                    tf.square(w))
                            )
                        + tf.square(
                            tf.matmul(
                                tf.multiply(Rms, Cms),
                                w)
                            ))
                    )
            Zi = tf.matmul(Le, w) + \
                tf.divide(
                    tf.multiply(
                        tf.matmul(
                            tf.multiply(BLsq, Cms),
                            w),
                        1 - tf.matmul(
                                 tf.multiply(Mms, Cms),
                                 tf.square(w)
                                 )
                        ),
                    (tf.square(
                            1 - tf.matmul(
                                    tf.multiply(Mms, Cms),
                                    tf.square(w))
                            )
                        + tf.square(
                            tf.matmul(
                                tf.multiply(Rms, Cms),
                                w)
                            ))
                    )
        return Zr, Zi

     def apply_z_scale(self, Re, Le, Rms, Cms, Mms, BLsq, x_scaler=None):
        if x_scaler is None:
            x_scaler = {'Rms_sigma': 1, 'Rms_mu': 0,
                        'Cms_sigma': 1, 'Cms_mu': 0,
                        'Mms_sigma': 1, 'Mms_mu': 0,
                        'BLsq_sigma': 1, 'BLsq_mu': 0,
                        'Re_sigma': 1, 'Re_mu': 0,
                        'Le_sigma': 1, 'Le_mu': 0}
        Rms = (Rms - x_scaler['Rms_mu']) / x_scaler['Rms_sigma']
        Cms = (Cms - x_scaler['Cms_mu']) / x_scaler['Cms_sigma']
        Mms = (Mms - x_scaler['Mms_mu']) / x_scaler['Mms_sigma']
        BLsq = (BLsq - x_scaler['BLsq_mu']) / x_scaler['BLsq_sigma']
        Re = (Re - x_scaler['Re_mu']) / x_scaler['Re_sigma']
        Le = (Le - x_scaler['Le_mu']) / x_scaler['Le_sigma']
        return Re, Le, Rms, Cms, Mms, BLsq

    def revert_z_scale(self, Re, Le, Rms, Cms, Mms, BLsq, x_scaler=None):
        if x_scaler is None:
            x_scaler = {'Rms_sigma': 1, 'Rms_mu': 0,
                        'Cms_sigma': 1, 'Cms_mu': 0,
                        'Mms_sigma': 1, 'Mms_mu': 0,
                        'BLsq_sigma': 1, 'BLsq_mu': 0,
                        'Re_sigma': 1, 'Re_mu': 0,
                        'Le_sigma': 1, 'Le_mu': 0}
        Rms = Rms * x_scaler['Rms_sigma'] + x_scaler['Rms_mu']
        Cms = Cms * x_scaler['Cms_sigma'] + x_scaler['Cms_mu']
        Mms = Mms * x_scaler['Mms_sigma'] + x_scaler['Mms_mu']
        BLsq = BLsq * x_scaler['BLsq_sigma'] + x_scaler['BLsq_mu']
        Re = Re * x_scaler['Re_sigma'] + x_scaler['Re_mu']
        Le = Le * x_scaler['Le_sigma'] + x_scaler['Le_mu']
        return Re, Le, Rms, Cms, Mms, BLsq

    def fit_physics(self, z_train, x_train, epochs=1, batch_size=10,
                    verbose=1, x_scaler=None, alpha_phy=None,
                    add_AE_loss=False, add_TS_loss=True, multi_step=False):
        # Convert all data into single dataset as Dataset dtype
        dataset = tf.data.Dataset.from_tensor_slices(
            (z_train, x_train)
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
            for z, x in dataset:  # Returns a single batch per loop iteration
                if multi_step:
                    train_metrics = self.train_step_multi(z, x, x_scaler, alpha_phy,
                                                          add_AE_loss, add_TS_loss)
                else:
                    train_metrics = self.train_step(z, x, x_scaler, alpha_phy,
                                                    add_AE_loss, add_TS_loss)
            end_time = time.time()
            T = end_time - start_time
            if (verbose == 1) or (verbose == 2):
                logs = ""
                for m in self.metrics:
                    try:
                        if train_metrics[m.name] < 0.0001:
                            logs = logs \
                                + "{:}: {:.4E} - ".format(
                                    m.name, train_metrics[m.name]
                                    )
                        else:
                            logs = logs + "{:}: {:.4f} - ".format(m.name,
                                                                  train_metrics[m.name])
                    except:
                        continue
                strA = "{:1d}/{:1d} - {:1g}s - ".format(
                        len(dataset), len(dataset), round(T)
                        )
                strZ = "{:1g}s/epoch - {:3g}ms/step".format(round(T),
                                                            round(1000*T/len(dataset)))
                print(strA, logs, strZ, sep="")
            # Populate history
            for m in self.metrics[:len(history)]:
                history[m.name].append(m.result().numpy())
                # Reset metric values
                m.reset_states()
        return history

    def train_step(self, Z, x, x_scaler=None, alpha_phy=None,
                   add_AE_loss=False, add_TS_loss=True):
        # Instantiate alpha_phy
        if alpha_phy is None:
            alpha_phy = np.ones((8, 1))
            
        # Define loss functions
        MSE_loss = tf.keras.losses.MeanSquaredError()
        Huber_loss = tf.keras.losses.Huber()
        
        # Define frequency array, 'omega', for tensorflow 
        omega = 2 * np.pi * np.linspace(1, Z.shape[1], Z.shape[1])
        w = tf.constant(omega)
        w = tf.reshape(w, [1, w.shape[0]])

        with tf.GradientTape(persistent=True) as tape:
            # Get encoder predictions
            x_pred = self.encoder(Z)
            # Other curve based parameters (e.g., w0)
            z_params = Z[:, (self.input_dim[0]-2):, 0]
            
            # --- Calculate losses ---
            # 1) Data Loss
            data_loss = tf.cast(MSE_loss(x, x_pred), dtype=tf.float32)

            # --- Physics-based loss terms ---
            if (add_AE_loss is True) or (add_TS_loss is True):
                # 2) Physics Loss
                # Get true and predicted Thiele-Small parameters
                # >>> TS by NN architecture output
                
                # >>> Parameters learned from Zr
                Rms_true, Rms_pred = x[:, 0], x_pred[:, 0]
                BLsq_true, BLsq_pred = x[:, 1], x_pred[:, 1]
                Re_true, Re_pred = x[:, 2], x_pred[:, 2]
                
                # >>> Parameters learned from Zi
                Mms_true, Mms_pred = x[:, 3], x_pred[:, 3]
                Cms_true, Cms_pred = x[:, 4], x_pred[:, 4]
                Le_true, Le_pred = x[:, 5], x_pred[:, 5]

                # --- Reshape the tensors ---
                # Using x_pred variables
                Rms_pred = tf.reshape(Rms_pred, shape=[Rms_pred.shape[0], 1])
                Cms_pred = tf.reshape(Cms_pred, shape=[Cms_pred.shape[0], 1])
                Mms_pred = tf.reshape(Mms_pred, shape=[Mms_pred.shape[0], 1])
                BLsq_pred = tf.reshape(BLsq_pred, shape=[BLsq_pred.shape[0], 1])
                Re_pred = tf.reshape(Re_pred, shape=[Re_pred.shape[0], 1])
                Le_pred = tf.reshape(Le_pred, shape=[Le_pred.shape[0], 1])
                # Using x_true variables
                Rms_true = tf.reshape(Rms_true, shape=[Rms_true.shape[0], 1])
                Cms_true = tf.reshape(Cms_true, shape=[Cms_true.shape[0], 1])
                Mms_true = tf.reshape(Mms_true, shape=[Mms_true.shape[0], 1])
                BLsq_true = tf.reshape(BLsq_true, shape=[BLsq_true.shape[0], 1])
                Re_true = tf.reshape(Re_true, shape=[Re_true.shape[0], 1])
                Le_true = tf.reshape(Le_true, shape=[Le_true.shape[0], 1])

                # --- Cast datatype ---
                Rms_pred = tf.cast(Rms_pred, dtype=tf.float64)
                Cms_pred = tf.cast(Cms_pred, dtype=tf.float64)
                Mms_pred = tf.cast(Mms_pred, dtype=tf.float64)
                BLsq_pred = tf.cast(BLsq_pred, dtype=tf.float64)
                Re_pred = tf.cast(Re_pred, dtype=tf.float64)
                Le_pred = tf.cast(Le_pred, dtype=tf.float64)

                # --- Rescale z-scores to actual values ---
                # True values
                Re_true, Le_true, \
                    Rms_true, Cms_true, \
                    Mms_true, BLsq_true = self.revert_z_scale(Re_true, Le_true,
                                                              Rms_true, Cms_true,
                                                              Mms_true, BLsq_true,
                                                              x_scaler=x_scaler)
                # Predicted values
                Re_pred, Le_pred, \
                    Rms_pred, Cms_pred, \
                    Mms_pred, BLsq_pred = self.revert_z_scale(Re_pred, Le_pred,
                                                              Rms_pred, Cms_pred,
                                                              Mms_pred, BLsq_pred,
                                                              x_scaler=x_scaler)

                # --- TS Parameter losses ---
                if add_TS_loss is True:
                    w0 = z_params[:, 0]

                    # --- Define frequency points for gradient calcs ---
                    # Rescale w0 
                    w0 = w0 * x_scaler['w0_sigma'] + x_scaler['w0_mu']
                    w0 = tf.reshape(w0, [w0.shape[0], 1])
                    # Define w_zero
                    w_zero = tf.ones_like(w0)
                    w_zero = tf.multiply(2*np.pi, w_zero)
                    # Define w_last
                    w_last = tf.ones_like(w0)
                    w_last = tf.multiply(2*np.pi*10000, w_last)

                    # --- Initialize variables ---
                    with tf.GradientTape(persistent=True) as outer_tape:
                        # Watch omega = 0, w_zero, and omega = omega0, w_0, to
                        # calculate respective gradients
                        outer_tape.watch(w_zero)
                        outer_tape.watch(w0)
                        outer_tape.watch(w_last)

                        with tf.GradientTape(persistent=True) as inner_tape:
                            inner_tape.watch(w_zero)
                            inner_tape.watch(w0)
                            inner_tape.watch(w_last)
                            # --- Calculate Zr, Zi ---
                            # >>> Using x_true - As if using input Z
                            
                            # w = 0
                            Zr_wz, Zi_wz = self.calc_Z(w_zero,
                                                       Re_true,
                                                       Le_true,
                                                       Cms_true,
                                                       Mms_true,
                                                       Rms_true,
                                                       BLsq_true)
                            # w = w0
                            Zr_w0, Zi_w0 = self.calc_Z(w0,
                                                       Re_true,
                                                       Le_true,
                                                       Cms_true,
                                                       Mms_true,
                                                       Rms_true,
                                                       BLsq_true)
                            # w = 0 -> For Re_hat calc
                            Zr_wz_pred, _ = self.calc_Z(w_zero,
                                                        Re_pred,
                                                        Le_true,
                                                        Cms_true,
                                                        Mms_true,
                                                        Rms_true,
                                                        BLsq_true)
                            # w >> w0 -> For Le_hat calc
                            _, Zi_wlast_pred = self.calc_Z(w_last,
                                                           Re_true,
                                                           Le_pred,
                                                           Cms_true,
                                                           Mms_true,
                                                           Rms_true,
                                                           BLsq_true)
                        # Calculate first derivatives
                        dZr_dw_wz = inner_tape.gradient(Zr_wz, w_zero)
                        dZi_dw_wz = inner_tape.gradient(Zi_wz, w_zero)
                        dZi_dw_w0 = inner_tape.gradient(Zi_w0, w0)
                        dZi_dw_wlast_pred = inner_tape.gradient(Zi_wlast_pred,
                                                                w_last)
                    # Calculate the second derivative
                    d2Zr_dw2_wz = outer_tape.gradient(dZr_dw_wz, w_zero)

                    # Calculate predicted Thiele-Small parameters from gradients
                    # >>> Final Calculations
                    Rms_hat = tf.multiply(tf.divide(np.sqrt(2.0), 2.0 * Cms_pred),
                                          tf.sqrt(tf.divide(d2Zr_dw2_wz,
                                                            (Zr_w0 - Zr_wz)))
                                          )
                    Cms_hat = tf.multiply(tf.divide(1.0, (2.0 * Rms_pred)),
                                          tf.divide(d2Zr_dw2_wz,
                                                    (dZi_dw_wz - Le_pred))
                                          )
                    Mms_hat = tf.divide(tf.multiply(BLsq_pred,
                                                    (Le_pred - dZi_dw_w0)),
                                        2.0 * tf.square((Zr_w0 - Zr_wz))
                                        )
                    BLsq_hat = tf.divide(d2Zr_dw2_wz,
                                          tf.multiply(2.0 * Rms_pred,
                                                      tf.square(Cms_pred))
                                          )
                    Re_hat = Zr_wz_pred
                    Le_hat = dZi_dw_wlast_pred

                    # --- Reapply z-score to parameters for loss calculations
                    # True values
                    Re_true, Le_true, \
                        Rms_true, Cms_true, \
                        Mms_true, BLsq_true = self.apply_z_scale(Re_true, Le_true,
                                                                 Rms_true, Cms_true,
                                                                 Mms_true, BLsq_true,
                                                                 x_scaler=x_scaler)
                    # Predicted values
                    Re_hat, Le_hat, \
                        Rms_hat, Cms_hat, \
                        Mms_hat, BLsq_hat = self.apply_z_scale(Re_hat, Le_hat,
                                                               Rms_hat, Cms_hat,
                                                               Mms_hat, BLsq_hat,
                                                               x_scaler=x_scaler)

                    # --- Physics Losses ---
                    Rms_loss = alpha_phy[0] * MSE_loss(Rms_true, Rms_hat)
                    BLsq_loss = alpha_phy[1] * MSE_loss(BLsq_true, BLsq_hat)
                    Re_loss = alpha_phy[2] * MSE_loss(Re_true, Re_hat)
                    Mms_loss = alpha_phy[3] * MSE_loss(Mms_true, Mms_hat)
                    Cms_loss = alpha_phy[4] * MSE_loss(Cms_true, Cms_hat)
                    Le_loss = alpha_phy[5] * MSE_loss(Le_true, Le_hat)

                    # --- Cast to same dtype ---
                    Rms_loss = tf.cast(Rms_loss, dtype=tf.float32)
                    BLsq_loss = tf.cast(BLsq_loss, dtype=tf.float32)
                    Re_loss = tf.cast(Re_loss, dtype=tf.float32)
                    Mms_loss = tf.cast(Mms_loss, dtype=tf.float32)
                    Cms_loss = tf.cast(Cms_loss, dtype=tf.float32)
                    Le_loss = tf.cast(Le_loss, dtype=tf.float32)

                # --- 'autoencoder' Loss ---
                if add_AE_loss is True:
                    # --- Calculate predicted Zr, Zi for all w given predicted
                    #     Thiele-Small parameters ---
                    Zr_pred, Zi_pred = self.calc_Z(w[:, :-2], Re_pred, Le_pred,
                                                   Cms_pred, Mms_pred, Rms_pred,
                                                   BLsq_pred)

                    Zr_loss = alpha_phy[6] * Huber_loss(Z[:, :-2, 0], Zr_pred)
                    Zi_loss = alpha_phy[7] * Huber_loss(Z[:, :-2, 1], Zi_pred)

                    # Cast to same dtype
                    Zr_loss = tf.cast(Zr_loss, dtype=tf.float32)
                    Zi_loss = tf.cast(Zi_loss, dtype=tf.float32)

                # --- Define total_loss ---
                if (add_AE_loss is True) and (add_TS_loss is True):
                    # Calculate physics_loss
                    physics_loss = Rms_loss + Cms_loss + Mms_loss \
                                 + BLsq_loss + Re_loss + Le_loss \
                                 + Zr_loss + Zi_loss
                    # Define total_loss as list of included losses
                    total_loss = [data_loss, Rms_loss, Cms_loss, Mms_loss,
                                  BLsq_loss, Re_loss, Le_loss,
                                  Zr_loss, Zi_loss]

                elif (add_AE_loss is True) and (add_TS_loss is False):
                    # Calculate physics_loss
                    physics_loss = Zr_loss + Zi_loss
                    # Define total_loss as list of included losses
                    total_loss = [data_loss, Zr_loss, Zi_loss]
                else:
                    # Calculate physics_loss
                    physics_loss = Rms_loss + Cms_loss + Mms_loss \
                                 + BLsq_loss + Re_loss + Le_loss
                    # Define total_loss as list of included losses
                    total_loss = [data_loss, Rms_loss, Cms_loss, Mms_loss,
                                  BLsq_loss, Re_loss, Le_loss]

                total_loss_value = sum(total_loss)
            else:
                total_loss = data_loss
                total_loss_value = total_loss
        # Get gradients and apply to optimizer
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Define dictionary of training metrics
        train_metrics = {}
        # --- Update metrics and add to dictionary ---
        if (add_AE_loss is True) or (add_TS_loss is True):
            # --- Data loss ---
            self.data_loss.update_state(data_loss)
            train_metrics['data_loss'] = self.metrics[1].result().numpy()
            # --- TS loss ---
            if add_TS_loss is True:
                self.Re_loss.update_state(Re_loss)
                self.Rms_loss.update_state(Rms_loss)
                self.Cms_loss.update_state(Cms_loss)
                self.Mms_loss.update_state(Mms_loss)
                self.BLsq_loss.update_state(BLsq_loss)
                self.Le_loss.update_state(Le_loss)
                train_metrics['Rms_loss'] = self.metrics[3].result().numpy()
                train_metrics['Cms_loss'] = self.metrics[4].result().numpy()
                train_metrics['Mms_loss'] = self.metrics[5].result().numpy()
                train_metrics['BLsq_loss'] = self.metrics[6].result().numpy()
                train_metrics['Re_loss'] = self.metrics[7].result().numpy()
                train_metrics['Le_loss'] = self.metrics[8].result().numpy()
            # --- AE loss ---
            if add_AE_loss is True:
                self.Zr_loss.update_state(Zr_loss)
                self.Zi_loss.update_state(Zi_loss)
                train_metrics['Zr_loss'] = self.metrics[9].result().numpy()
                train_metrics['Zi_loss'] = self.metrics[10].result().numpy()
            # --- Total physics loss ---
            self.physics_loss.update_state(physics_loss)
            train_metrics['physics_loss'] = self.metrics[2].result().numpy()
        # --- Total loss ---
        self.total_loss.update_state(total_loss_value)
        train_metrics['total_loss'] = self.metrics[0].result().numpy()
        return train_metrics

    def train_step_multi(self, Z, x, x_scaler=None, alpha_phy=None,
                         add_AE_loss=False, add_TS_loss=True):
        # Instantiate alpha_phy
        if alpha_phy is None:
            alpha_phy = np.ones((8, 1))
            
        # Define loss objects
        MSE_loss = tf.keras.losses.MeanSquaredError()
        Huber_loss = tf.keras.losses.Huber()
        
        # Define frequency array, 'omega', for tensorflow
        omega = 2 * np.pi * np.linspace(1, Z.shape[1], Z.shape[1])
        w = tf.constant(omega)
        w = tf.reshape(w, [1, w.shape[0]])

        # Get trainable layer prefixes
        prefixes = ['Rms', 'Cms', 'Mms', 'BLsq', 'Re', 'Le']
        TS_training = []
        for prefix in prefixes:
            for lay in self.encoder.layers:
                if (lay.trainable is True) and (lay.name.startswith(prefix)):
                    if prefix not in TS_training:
                        TS_training.append(prefix)

        # Instantiate data_losses
        data_losses = []
        # Calculate gradients for model weights
        with tf.GradientTape(persistent=True) as tape:
            # Get encoder predictions
            x_pred = self.encoder(Z)
            # Other curve based parameters (e.g., w0)
            z_params = Z[:, (self.input_dim[0]-2):, 0]

            # Get true and predicted Thiele-Small parameters
            Rms_true, Rms_pred = x[:, 0], x_pred[:, 0]
            BLsq_true, BLsq_pred = x[:, 1], x_pred[:, 1]
            Re_true, Re_pred = x[:, 2], x_pred[:, 2]
            Mms_true, Mms_pred = x[:, 3], x_pred[:, 3]
            Cms_true, Cms_pred = x[:, 4], x_pred[:, 4]
            Le_true, Le_pred = x[:, 5], x_pred[:, 5]

            if 'Rms' in TS_training:
                data_losses.append(tf.cast(MSE_loss(Rms_true, Rms_pred),
                                           dtype=tf.float32))
            if 'Cms' in TS_training:
                data_losses.append(tf.cast(MSE_loss(Cms_true, Cms_pred),
                                           dtype=tf.float32))
            if 'Mms' in TS_training:
                data_losses.append(tf.cast(MSE_loss(Mms_true, Mms_pred),
                                           dtype=tf.float32))
            if 'BLsq' in TS_training:
                data_losses.append(tf.cast(MSE_loss(BLsq_true, BLsq_pred),
                                           dtype=tf.float32))
            if 'Re' in TS_training:
                data_losses.append(tf.cast(MSE_loss(Re_true, Re_pred),
                                           dtype=tf.float32))
            if 'Le' in TS_training:
                data_losses.append(tf.cast(MSE_loss(Le_true, Le_pred),
                                           dtype=tf.float32))
            # --- Calculate losses ---
            # 1) Data Loss
            data_loss = sum(data_losses)

            if (add_AE_loss is True) or (add_TS_loss is True):
                # --- 2) Physics Loss ------------------------
                # --- Reshape the tensors ---
                # Using x_pred variables
                Rms_pred = tf.reshape(Rms_pred, shape=[Rms_pred.shape[0], 1])
                Cms_pred = tf.reshape(Cms_pred, shape=[Cms_pred.shape[0], 1])
                Mms_pred = tf.reshape(Mms_pred, shape=[Mms_pred.shape[0], 1])
                BLsq_pred = tf.reshape(BLsq_pred, shape=[BLsq_pred.shape[0], 1])
                Re_pred = tf.reshape(Re_pred, shape=[Re_pred.shape[0], 1])
                Le_pred = tf.reshape(Le_pred, shape=[Le_pred.shape[0], 1])
                # Using x_true variables
                Rms_true = tf.reshape(Rms_true, shape=[Rms_true.shape[0], 1])
                Cms_true = tf.reshape(Cms_true, shape=[Cms_true.shape[0], 1])
                Mms_true = tf.reshape(Mms_true, shape=[Mms_true.shape[0], 1])
                BLsq_true = tf.reshape(BLsq_true, shape=[BLsq_true.shape[0], 1])
                Re_true = tf.reshape(Re_true, shape=[Re_true.shape[0], 1])
                Le_true = tf.reshape(Le_true, shape=[Le_true.shape[0], 1])

                # --- Cast datatype ---
                Rms_pred = tf.cast(Rms_pred, dtype=tf.float64)
                Cms_pred = tf.cast(Cms_pred, dtype=tf.float64)
                Mms_pred = tf.cast(Mms_pred, dtype=tf.float64)
                BLsq_pred = tf.cast(BLsq_pred, dtype=tf.float64)
                Re_pred = tf.cast(Re_pred, dtype=tf.float64)
                Le_pred = tf.cast(Le_pred, dtype=tf.float64)

                # --- Rescale z-scores to actual values ---
                # True values
                Re_true, Le_true, \
                    Rms_true, Cms_true, \
                    Mms_true, BLsq_true = self.revert_z_scale(Re_true, Le_true,
                                                              Rms_true, Cms_true,
                                                              Mms_true, BLsq_true,
                                                              x_scaler=x_scaler)
                # Predicted values
                Re_pred, Le_pred, \
                    Rms_pred, Cms_pred, \
                    Mms_pred, BLsq_pred = self.revert_z_scale(Re_pred, Le_pred,
                                                              Rms_pred, Cms_pred,
                                                              Mms_pred, BLsq_pred,
                                                              x_scaler=x_scaler)

                # ---TS Parameter losses ---
                if add_TS_loss is True:
                    # Get, rescale, and reshape w0
                    w0 = z_params[:, 0]
                    w0 = w0 * x_scaler['w0_sigma'] + x_scaler['w0_mu']
                    w0 = tf.reshape(w0, [w0.shape[0], 1])
                    # Define w_zero
                    w_zero = tf.ones_like(w0)
                    w_zero = tf.multiply(2*np.pi, w_zero)
                    # Define w_last
                    w_last = tf.ones_like(w0)
                    w_last = tf.multiply(2*np.pi*10000, w_last)

                    # --- Initialize variables ---
                    with tf.GradientTape(persistent=True) as outer_tape:
                        # Watch omega = 0, w_zero, and omega = omega0, w_0, to
                        # calculate respective gradients
                        outer_tape.watch(w_zero)
                        outer_tape.watch(w0)
                        outer_tape.watch(w_last)

                        with tf.GradientTape(persistent=True) as inner_tape:
                            inner_tape.watch(w_zero)
                            inner_tape.watch(w0)
                            inner_tape.watch(w_last)
                            # --- Calculate Zr, Zi ---
                            # w = 0
                            Zr_wz, Zi_wz = self.calc_Z(w_zero,
                                                       Re_true,
                                                       Le_true,
                                                       Cms_true,
                                                       Mms_true,
                                                       Rms_true,
                                                       BLsq_true)
                            # w = w0
                            Zr_w0, Zi_w0 = self.calc_Z(w0,
                                                       Re_true,
                                                       Le_true,
                                                       Cms_true,
                                                       Mms_true,
                                                       Rms_true,
                                                       BLsq_true)
                            # w = 0 -> For Re_hat calc
                            Zr_wz_pred, _ = self.calc_Z(w_zero,
                                                        Re_pred,
                                                        Le_true,
                                                        Cms_true,
                                                        Mms_true,
                                                        Rms_true,
                                                        BLsq_true)
                            # w >> w0 -> For Le_hat calc
                            _, Zi_wlast_pred = self.calc_Z(w_last,
                                                           Re_true,
                                                           Le_pred,
                                                           Cms_true,
                                                           Mms_true,
                                                           Rms_true,
                                                           BLsq_true)
                        # Calculate first derivatives
                        dZr_dw_wz = inner_tape.gradient(Zr_wz, w_zero)
                        dZi_dw_wz = inner_tape.gradient(Zi_wz, w_zero)
                        dZi_dw_w0 = inner_tape.gradient(Zi_w0, w0)
                        dZi_dw_wlast_pred = inner_tape.gradient(Zi_wlast_pred,
                                                                w_last)
                    # Calculate the second derivative
                    d2Zr_dw2_wz = outer_tape.gradient(dZr_dw_wz, w_zero)

                    # Calculate predicted Thiele-Small parameters from gradients
                    Rms_hat = tf.multiply(tf.divide(np.sqrt(2.0), 2.0 * Cms_pred),
                                          tf.sqrt(tf.divide(d2Zr_dw2_wz,
                                                            (Zr_w0 - Zr_wz)))
                                          )
                    Cms_hat = tf.multiply(tf.divide(1.0, (2.0 * Rms_pred)),
                                          tf.divide(d2Zr_dw2_wz,
                                                    (dZi_dw_wz - Le_pred))
                                          )
                    Mms_hat = tf.divide(tf.multiply(BLsq_pred,
                                                    (Le_pred - dZi_dw_w0)),
                                        2.0 * tf.square((Zr_w0 - Zr_wz))
                                        )
                    BLsq_hat = tf.divide(d2Zr_dw2_wz,
                                         tf.multiply(2.0 * Rms_pred,
                                                     tf.square(Cms_pred))
                                         )
                    Re_hat = Zr_wz_pred
                    Le_hat = dZi_dw_wlast_pred

                    # --- Reapply z-score to parameters for loss calculations ---
                    # True values
                    Re_true, Le_true, \
                        Rms_true, Cms_true, \
                        Mms_true, BLsq_true = self.apply_z_scale(Re_true, Le_true,
                                                                 Rms_true, Cms_true,
                                                                 Mms_true, BLsq_true,
                                                                 x_scaler=x_scaler)
                    # Predicted values
                    Re_hat, Le_hat, \
                        Rms_hat, Cms_hat, \
                        Mms_hat, BLsq_hat = self.apply_z_scale(Re_hat, Le_hat,
                                                               Rms_hat, Cms_hat,
                                                               Mms_hat, BLsq_hat,
                                                               x_scaler=x_scaler)

                    # --- Physics Losses (scaled by alpha_phy) ---
                    Rms_loss = alpha_phy[0] * MSE_loss(Rms_true, Rms_hat)
                    BLsq_loss = alpha_phy[1] * MSE_loss(BLsq_true, BLsq_hat)
                    Re_loss = alpha_phy[2] * MSE_loss(Re_true, Re_hat)
                    Mms_loss = alpha_phy[3] * MSE_loss(Mms_true, Mms_hat)
                    Cms_loss = alpha_phy[4] * MSE_loss(Cms_true, Cms_hat)
                    Le_loss = alpha_phy[5] * MSE_loss(Le_true, Le_hat)

                    # --- Cast to same dtype ---
                    Rms_loss = tf.cast(Rms_loss, dtype=tf.float32)
                    BLsq_loss = tf.cast(BLsq_loss, dtype=tf.float32)
                    Re_loss = tf.cast(Re_loss, dtype=tf.float32)
                    Mms_loss = tf.cast(Mms_loss, dtype=tf.float32)
                    Cms_loss = tf.cast(Cms_loss, dtype=tf.float32)
                    Le_loss = tf.cast(Le_loss, dtype=tf.float32)

                # Instantiate losses
                physics_loss = []
                total_loss = [data_loss]
                if add_AE_loss is True:
                    # --- Calculate predicted Zr, Zi for all w given predicted
                    if 'Rms' in TS_training:
                        Rms_input = Rms_pred
                    else:
                        Rms_input = Rms_true
                    if 'Cms' in TS_training:
                        Cms_input = Cms_pred
                    else:
                        Cms_input = Cms_true
                    if 'Mms' in TS_training:
                        Mms_input = Mms_pred
                    else:
                        Mms_input = Mms_true
                    if 'BLsq' in TS_training:
                        BLsq_input = BLsq_pred
                    else:
                        BLsq_input = BLsq_true
                    if 'Re' in TS_training:
                        Re_input = Re_pred
                    else:
                        Re_input = Re_true
                    if 'Le' in TS_training:
                        Le_input = Le_pred
                    else:
                        Le_input = Le_true

                    # --- AE Loss -----------------------------------------------
                    # --- Calculate Z(w) w/ input Thiele-Small parameters ---
                    Zr_pred, Zi_pred = self.calc_Z(w[:, :-2], Re_input, Le_input,
                                                   Cms_input, Mms_input,
                                                   Rms_input, BLsq_input)
                    # Calculate loss based on if Zmag is True or False
                    if self.Zmag is True:
                        Zt_pred = tf.sqrt(tf.square(Zr_pred) +
                                          tf.square(Zi_pred))
                        Zr_loss = alpha_phy[6] * Huber_loss(Z[:, :-2, 0], Zt_pred)
                        Zi_loss = tf.constant(0.0)
                    else:
                        Zr_loss = alpha_phy[6] * Huber_loss(Z[:, :-2, 0], Zr_pred)
                        Zi_loss = alpha_phy[7] * Huber_loss(Z[:, :-2, 1], Zi_pred)

                    # Cast to same dtype
                    Zr_loss = tf.cast(Zr_loss, dtype=tf.float32)
                    Zi_loss = tf.cast(Zi_loss, dtype=tf.float32)

                if (add_AE_loss is True) and (add_TS_loss is True):
                    # Define total_loss as list of included losses
                    if 'Rms' in TS_training:
                        physics_loss.append(Rms_loss)
                        total_loss.append(Rms_loss)
                    if 'Cms' in TS_training:
                        physics_loss.append(Cms_loss)
                        total_loss.append(Cms_loss)
                    if 'Mms' in TS_training:
                        physics_loss.append(Mms_loss)
                        total_loss.append(Mms_loss)
                    if 'BLsq' in TS_training:
                        physics_loss.append(BLsq_loss)
                        total_loss.append(BLsq_loss)
                    if 'Re' in TS_training:
                        physics_loss.append(Re_loss)
                        total_loss.append(Re_loss)
                    if 'Le' in TS_training:
                        physics_loss.append(Le_loss)
                        total_loss.append(Le_loss)
                    physics_loss.append(Zr_loss)
                    physics_loss.append(Zi_loss)
                    total_loss.append(Zr_loss)
                    total_loss.append(Zi_loss)

                elif (add_AE_loss is True) and (add_TS_loss is False):
                    # Calculate physics_loss
                    physics_loss = [Zr_loss, Zi_loss]
                    # Define total_loss as list of included losses
                    total_loss = [data_loss, Zr_loss, Zi_loss]
                else:
                    # Calculate physics_loss
                    if 'Rms' in TS_training:
                        physics_loss.append(Rms_loss)
                        total_loss.append(Rms_loss)
                    if 'Cms' in TS_training:
                        physics_loss.append(Cms_loss)
                        total_loss.append(Cms_loss)
                    if 'Mms' in TS_training:
                        physics_loss.append(Mms_loss)
                        total_loss.append(Mms_loss)
                    if 'BLsq' in TS_training:
                        physics_loss.append(BLsq_loss)
                        total_loss.append(BLsq_loss)
                    if 'Re' in TS_training:
                        physics_loss.append(Re_loss)
                        total_loss.append(Re_loss)
                    if 'Le' in TS_training:
                        physics_loss.append(Le_loss)
                        total_loss.append(Le_loss)

                physics_loss_value = sum(physics_loss)
                total_loss_value = sum(total_loss)
            else:
                total_loss = data_loss
                total_loss_value = total_loss
        # Get gradients and apply to optimizer
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Define dictionary of training metrics
        train_metrics = {}
        # --- Update metrics and add to dictionary ---
        if (add_AE_loss is True) or (add_TS_loss is True):
            # --- Data loss ---
            self.data_loss.update_state(data_loss)
            train_metrics['data_loss'] = self.metrics[1].result().numpy()
            # --- TS loss ---
            if add_TS_loss is True:
                if 'Rms' in TS_training:
                    # self.Rms_loss.update_state(Rms_loss)
                    self.Rms_loss.update_state(MSE_loss(Rms_true, Rms_hat))
                    train_metrics['Rms_loss'] = self.metrics[3].result().numpy()
                if 'Cms' in TS_training:
                    # self.Cms_loss.update_state(Cms_loss)
                    self.Cms_loss.update_state(MSE_loss(Cms_true, Cms_hat))
                    train_metrics['Cms_loss'] = self.metrics[4].result().numpy()
                if 'Mms' in TS_training:
                    # self.Mms_loss.update_state(Mms_loss)
                    self.Mms_loss.update_state(MSE_loss(Mms_true, Mms_hat))
                    train_metrics['Mms_loss'] = self.metrics[5].result().numpy()
                if 'BLsq' in TS_training:
                    # self.BLsq_loss.update_state(BLsq_loss)
                    self.BLsq_loss.update_state(MSE_loss(BLsq_true, BLsq_hat))
                    train_metrics['BLsq_loss'] = self.metrics[6].result().numpy()
                if 'Re' in TS_training:
                    # self.Re_loss.update_state(Re_loss)
                    self.Re_loss.update_state(MSE_loss(Re_true, Re_hat))
                    train_metrics['Re_loss'] = self.metrics[7].result().numpy()
                if 'Le' in TS_training:
                    # self.Le_loss.update_state(Le_loss)
                    self.Le_loss.update_state(MSE_loss(Le_true, Le_hat))
                    train_metrics['Le_loss'] = self.metrics[8].result().numpy()
            # --- AE Loss ---
            if add_AE_loss is True:
                self.Zr_loss.update_state(Zr_loss)
                self.Zi_loss.update_state(Zi_loss)
                train_metrics['Zr_loss'] = self.metrics[9].result().numpy()
                train_metrics['Zi_loss'] = self.metrics[10].result().numpy()
            # --- Total physics loss ---
            self.physics_loss.update_state(physics_loss_value)
            train_metrics['physics_loss'] = self.metrics[2].result().numpy()
        # --- Total loss ---
        # self.total_loss.update_state(total_loss)
        self.total_loss.update_state(total_loss_value)
        train_metrics['total_loss'] = self.metrics[0].result().numpy()
        return train_metrics

    def predict(self, Z):
        x_pred = self.encoder.predict(Z)
        return x_pred

    def save_model(self, modelName, history=None, save_path=os.getcwd()):
        # --- Save history data as .mat file ---
        if type(history) is dict:
            savemat(os.path.join(
                save_path,
                "History_" + modelName + ".mat"), history)
        else:
            print('---No history data saved for this model---')
        # --- Save weights by model ---
        # Model A
        self.encoder.save_weights(
            os.path.join(save_path, "Weights_" + modelName + "_Encoder.h5")
            )
        # Model B
        try:
            self.decoder.save_weights(
                os.path.join(save_path, "Weights_" + modelName + "_Decoder.h5")
                )
        except AttributeError:
            print('---No decoder defined---')

    def load_trained_weights(self, encoderFile, decoderFile=None):
        # --- Load weights .h5 files of each sub model ---
        # Model A
        self.encoder.load_weights(encoderFile)
        if decoderFile:
            # Model B
            self.decoder.load_weights(decoderFile)

    def set_layer_trainable(self, layer_name_prefix, trainable=True):
        # Iterate through each prefix
        for prefix in layer_name_prefix:
            # Iterate through each model layer
            for lay in self.encoder.layers:
                # Find layer with name prefix that matches prefix
                if lay.name.startswith(prefix):
                    # Set layer.trainable to either True or False
                    if trainable is True:
                        lay.trainable = True
                    else:
                        lay.trainable = False
