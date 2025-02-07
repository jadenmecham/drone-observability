import sys
import os

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras_visualizer import visualizer
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.pardir, 'util'))

import util
import figure_functions as ff


# from tensorflow.keras.util import get_custom_objects, register_keras_serializable
# get_custom_objects().clear()


class WindObserver:
    def __init__(self, dataset, input_names=None, output_names=None, time_steps=1, test_size=0.2, alpha=0.0):
        """ Train a neural network to predict wind direction/magnitude.

            Inputs:
                dataset:

        """

        # Store dataset
        self.dataset = dataset.copy()

        # Wrap angles
        self.dataset['zeta'] = util.wrapToPi(self.dataset['zeta'])  # wind direction
        self.dataset['psi'] = util.wrapToPi(self.dataset['psi'])  # heading
        self.dataset['beta'] = util.wrapToPi(self.dataset['beta'])  # course direction
        self.dataset['gamma'] = util.wrapToPi(self.dataset['gamma'])  # apparent wind direction
        self.dataset['alpha'] = util.wrapToPi(self.dataset['alpha'])  # acceleration angle

        # Inputs
        self.input_names = input_names
        self.output_names = output_names
        self.time_steps = time_steps
        self.test_size = test_size
        self.alpha = alpha

        self.n_input = len(self.input_names) * self.time_steps
        self.n_output = len(self.output_names)

        # Separate individual trajectories
        self.traj_list, self.n_traj = segment_from_df(self.dataset, column='time', val=0.0)

        # self.traj_start = np.where(self.dataset.time.values == 0.0)[0]  # where trajectories start
        # self.n_traj = self.traj_start.shape[0]  # number of trajectories

        # self.traj_list = []  # list of individual trajectories
        # for n in range(self.n_traj):
        #     if n == (self.n_traj - 1):
        #         traj_end = self.dataset.shape[0]
        #     else:
        #         traj_end = self.traj_start[n + 1]
        #
        #     traj = self.dataset.iloc[self.traj_start[n]:traj_end, :]
        #     self.traj_list.append(traj)

        # Train/test split
        self.train_size = 0.0
        self.train_index = np.array(0.0)
        self.test_index = np.array(0.0)
        self.train_traj = []
        self.test_traj = []
        self.train_test_split()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_train_noise = None
        self.X_test_noise = None

        self.y_train_predict = np.array(0.0)
        self.y_test_predict = np.array(0.0)

        self.y_train_noise_predict = np.array(0.0)
        self.y_test_noise_predict = np.array(0.0)

        # Augment data
        self.traj_list_augment = []
        self.train_traj_augment = []
        self.test_traj_augment = []
        self.augment_data()

        # Model
        self.model_type = None
        self.model = None
        self.loss = ''
        self.loss_function = None
        self.noise = None
        self.model_data = None
        self.predict = {}
        self.batch_size = 512
        self.epochs = 100

    def train_test_split(self):
        """ Split the dataset trajectories into train & test sets.
        """

        self.train_size = 1.0 - self.test_size
        traj_index = np.arange(0, self.n_traj)
        self.train_index, self.test_index = train_test_split(traj_index,
                                                             test_size=self.test_size,
                                                             train_size=self.train_size,
                                                             random_state=1,
                                                             shuffle=True,
                                                             stratify=None)

        self.train_traj = [self.traj_list[n] for n in self.train_index]
        self.test_traj = [self.traj_list[n] for n in self.test_index]

    def augment_data(self):
        """ Augment the data with prior time-steps.
        """

        self.traj_list_augment = []
        for traj in self.traj_list:
            traj_augment = util.collect_offset_rows(traj,
                                                    aug_column_names=self.input_names,
                                                    keep_column_names=self.output_names,
                                                    w=self.time_steps,
                                                    direction='backward')

            self.traj_list_augment.append(traj_augment)

        # self.traj_list_augment_all = pd.concat(self.traj_list_augment, ignore_index=True)

        # Separate the augmented train & test sets
        self.train_traj_augment = [self.traj_list_augment[n] for n in self.train_index]
        self.test_traj_augment = [self.traj_list_augment[n] for n in self.test_index]

        # Concatenate augmented train & test sets
        self.X_train = pd.concat(self.train_traj_augment, ignore_index=True)
        self.X_train = self.X_train.iloc[:, 0:self.n_input]

        self.y_train = pd.concat(self.train_traj_augment, ignore_index=True)
        self.y_train = self.y_train.iloc[:, self.n_input:]

        self.X_test = pd.concat(self.test_traj_augment, ignore_index=True)
        self.X_test = self.X_test.iloc[:, 0:self.n_input]

        self.y_test = pd.concat(self.test_traj_augment, ignore_index=True)
        self.y_test = self.y_test.iloc[:, self.n_input:]

    def define_model(self, model_type=None, loss='mse', noise=0.0):
        """ Define the neural network model.
        """

        self.loss = loss
        self.model_type = model_type
        self.noise = noise

        # Set model structure
        if self.model_type is None:
            self.model = Sequential()
            self.model.add(Dense(50, input_dim=self.n_input, activation='relu'))
            self.model.add(Dense(50, activation='relu'))
            self.model.add(Dense(20, activation='relu'))
            self.model.add(Dense(self.n_output, activation='linear'))

        elif isinstance(self.model_type, str):
            self.model = Sequential()
            self.model.add(tf.keras.layers.GaussianNoise(noise, seed=2))
            if self.model_type == 'v0':
                self.model.add(Dense(50, activation='relu', input_dim=self.n_input))
                self.model.add(Dense(50, activation='relu'))
                self.model.add(Dense(20, activation='relu'))
                self.model.add(Dense(self.n_output, activation='linear'))
            elif self.model_type == 'v1':
                self.model.add(Dense(50, activation='relu', input_dim=self.n_input))
                self.model.add(Dense(100, activation='relu'))
                self.model.add(Dense(50, activation='relu'))
                self.model.add(Dense(self.n_output, activation='linear'))

            elif self.model_type == 'v2':
                self.model.add(Dense(50, activation='relu', input_dim=self.n_input))
                self.model.add(Dense(100, activation='relu'))
                # self.model.add(Dense(100, activation='relu'))
                self.model.add(Dense(20, activation='relu'))
                self.model.add(Dense(self.n_output, activation='linear'))

        elif isinstance(self.model_type, tuple):  # model layers sizes are specified in tuple or list
            self.model = Sequential()
            self.model.add(tf.keras.layers.GaussianNoise(noise, seed=2))  # noise layer for training
            for k, layer_size in enumerate(self.model_type):
                if k == 0:  # input layer
                    self.model.add(Dense(layer_size, activation='relu', input_dim=self.n_input))
                else:
                    self.model.add(Dense(layer_size, activation='relu'))

            self.model.add(Dense(self.n_output, activation='linear'))  # output layer

        else:
            self.model = model_type

        # Set loss function
        if self.loss == 'circular':
            self.loss_function = self.custom_loss_circular
        else:
            self.loss_function = loss

        # Compile model
        self.model.compile(loss=self.loss_function, optimizer='adam')

    # @register_keras_serializable(package="my_package", name="custom_loss_circular")
    def custom_loss_circular(self, y_true, y_pred):
        """ Custom loss function for Keras neural network that handles a circular output variable.
        """

        # Transform into sine and cosine components
        y_true_sin = K.sin(y_true)
        y_true_cos = K.cos(y_true)

        y_pred_sin = K.sin(y_pred)
        y_pred_cos = K.cos(y_pred)

        # Calculate the MSE for each output
        loss_sin = K.mean(K.square(y_true_sin - y_pred_sin), axis=-1)
        loss_cos = K.mean(K.square(y_true_cos - y_pred_cos), axis=-1)

        # Calculate the trigonometric penalty
        trig_penalty = K.square(y_pred_sin) + K.square(y_true_cos) - 1.0
        penalty = K.mean(K.square(trig_penalty), axis=-1)

        # Combine the losses and add the penalty
        total_loss = loss_sin + loss_cos + self.alpha * penalty

        return total_loss

    def train_model(self, batch_size=256, epochs=100, verbose=1):
        """ Train the neural network model.
        """

        # print(device_lib.list_local_devices())
        print('GPU:', tf.config.list_physical_devices('GPU'))

        if batch_size is not None:
            self.batch_size = batch_size

        if epochs is not None:
            self.epochs = epochs

        self.model_data = self.model.fit(self.X_train.values, self.y_train.values,
                                         epochs=self.epochs,
                                         batch_size=self.batch_size,
                                         validation_split=0.2,
                                         verbose=verbose,
                                         shuffle=True)

    def run_model(self, noise=0.0, batch_size=4096):
        """ Run the neural network model.
        """

        # Run model on train & test sets
        X_train_noise = self.X_train + np.random.normal(loc=0.0, scale=noise, size=self.X_train.shape)
        self.y_train_predict = self.model.predict(X_train_noise, batch_size=batch_size).squeeze()

        X_test_noise = self.X_test + np.random.normal(loc=0.0, scale=noise, size=self.X_test.shape)
        self.y_test_predict = self.model.predict(X_test_noise, batch_size=batch_size).squeeze()

        # Compute error metrics
        self.predict = {'train': {}, 'test': {}}
        self.predict['train']['ground_truth'] = self.y_train.squeeze()
        self.predict['test']['ground_truth'] = self.y_test.squeeze()
        if self.loss == 'circular':
            self.predict['train']['predictions'] = util.wrapToPi(self.y_train_predict.squeeze())
            self.predict['train']['error'] = util.wrapToPi(
                self.predict['train']['predictions'] - self.predict['train']['ground_truth'])
            self.predict['train']['mean_error'] = scipy.stats.circmean(self.predict['train']['error'], high=np.pi,
                                                                       low=-np.pi)
            self.predict['train']['std_error'] = scipy.stats.circstd(self.predict['train']['error'], high=np.pi,
                                                                     low=-np.pi)

            self.predict['test']['predictions'] = util.wrapToPi(self.y_test_predict)
            self.predict['test']['error'] = util.wrapToPi(
                self.predict['test']['predictions'] - self.predict['test']['ground_truth'])
            self.predict['test']['mean_error'] = scipy.stats.circmean(self.predict['test']['error'], high=np.pi,
                                                                      low=-np.pi)
            self.predict['test']['std_error'] = scipy.stats.circstd(self.predict['test']['error'], high=np.pi,
                                                                    low=-np.pi)
        else:
            self.predict['train']['predictions'] = self.y_train_predict.copy()
            self.predict['train']['error'] = self.predict['train']['predictions'] - self.predict['train'][
                'ground_truth']
            self.predict['train']['mean_error'] = np.mean(self.predict['train']['error'])
            self.predict['train']['std_error'] = np.std(self.predict['train']['error'])

            self.predict['test']['predictions'] = self.y_test_predict.copy()
            self.predict['test']['error'] = self.predict['test']['predictions'] - self.predict['test']['ground_truth']
            self.predict['test']['mean_error'] = np.mean(self.predict['test']['error'])
            self.predict['test']['std_error'] = np.std(self.predict['test']['error'])

    def run_predict(self, traj_df, augment=True, noise=0.0, batch_size=4096, verbose=2):
        """ Run the neural network model.
        """

        traj_df = traj_df.copy().reset_index(drop=True)

        if augment:
            traj_augment = util.collect_offset_rows(traj_df,
                                                    aug_column_names=self.input_names,
                                                    keep_column_names=self.output_names,
                                                    w=self.time_steps,
                                                    direction='backward')
        else:
            traj_augment = traj_df

        # Get inputs & outputs
        X = traj_augment.iloc[:, 0:self.n_input]
        y = traj_augment.iloc[:, self.n_input:]

        # Add noise
        np.random.seed(seed=2)
        X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)

        # Run model on new data
        y_predict = self.model.predict(X, batch_size=batch_size, verbose=verbose).squeeze()

        # Aligned trajectory df
        n = int(np.floor(self.time_steps / 2))
        if (self.time_steps / 2) > np.floor(self.time_steps / 2):
            add_one = 0
        else:
            add_one = 1

        traj_df_aligned = traj_df.iloc[(n - add_one):traj_df.shape[0] - n, :].copy()
        predict_var = self.output_names[0] + str('_predict')
        traj_df_aligned.loc[:, predict_var] = y_predict.squeeze()

        return y_predict, y, traj_df_aligned

    def save(self, path):
        """ Save model and data.
        """

        # Make directory to save model
        if isinstance(self.model_type, tuple):
            model_label = '_'.join(map(str, self.model_type))
        else:
            model_label = self.model_type

        model_dir = (('network_inputs=' +
                      '_'.join(self.input_names) +
                      '_outputs=' + '_'.join(self.output_names) +
                      '_timesteps=' + str(self.time_steps)) +
                     '_noise=' + str(np.round(self.noise, 2)) +
                     '_model=' + model_label)

        model_path = os.path.join(path, model_dir)

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # Save model
        model_path_name = os.path.join(model_path, 'network.keras')
        self.model.save(model_path_name)

        # Save data
        save_path = os.path.join(model_path, 'model_data.pk')
        pkdata = {'traj_data': self.traj_list,
                  'traj_augment': self.traj_list_augment,
                  'n_traj': self.n_traj,
                  'input_names': self.input_names,
                  'output_names': self.output_names,
                  'time_steps': self.time_steps,
                  'loss': self.loss,
                  'model_type': self.model_type,
                  'train_index': self.train_index,
                  'test_index': self.test_index,
                  'X_train': self.X_train,
                  'y_train': self.y_train,
                  'y_train_predict': self.y_train_predict,
                  'X_test': self.X_test,
                  'y_test': self.y_test,
                  'y_test_predict': self.y_test_predict,
                  'history': self.model_data.history,
                  'params': self.model_data.params,
                  'predict': self.predict}

        with open(save_path, 'wb') as handle:
            pickle.dump(pkdata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_loss(self):
        fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=100)
        ax.plot(self.model_data.history['loss'], '-', label='loss', color='blue', linewidth=1.0)
        ax.plot(self.model_data.history['val_loss'], '-', label='val_loss', color='red', linewidth=1.0)
        ax.set_yscale('log')
        # ax.legend(bbox_to_anchor=(2.0, 1.0), fontsize=7)

    def plot_train_test(self):
        """ Plot train & test accuracy.
        """

        keys = self.predict.keys()
        n_key = len(keys)

        if self.output_names == 'zeta':
            bins = np.arange(-np.pi, np.pi, np.pi / 64)
        else:
            bins = 128

        fig, ax = plt.subplots(4, n_key, figsize=(2 * n_key, 4 * 2), dpi=100)

        cmap = ff.make_color_map(
            color_list=['white', 'navy', 'cornflowerblue', 'aqua', 'yellow', 'red', 'darkred'], N=256)

        for n, k in enumerate(keys):
            ax[0, n].set_title(k, fontsize=9, fontweight='bold')

            ax[0, n].plot(self.predict[k]['ground_truth'], self.predict[k]['predictions'], '.',
                          markersize=1, alpha=0.1, color='blueviolet')

            h = ax[1, n].hist2d(self.predict[k]['ground_truth'], self.predict[k]['predictions'],
                                bins=(bins, bins), cmap=cmap)

            if self.loss == 'circular':
                true_line = np.arange(-np.pi, np.pi, np.pi / 64)
                ax[0, n].plot(true_line, true_line, color='black', alpha=0.5, linewidth=1.0)
                mean = '$' + str(np.round(np.rad2deg(self.predict[k]['mean_error']), 2)) + '^\circ$'
                std = '$' + str(np.round(np.rad2deg(self.predict[k]['std_error']), 2)) + '^\circ$'
            else:
                true_line = np.arange(0.0, 1.0, 0.1)
                ax[0, n].plot(true_line, true_line, color='black', alpha=0.5, linewidth=1.0)
                mean = '$' + str(np.round(self.predict[k]['mean_error'], 2)) + 'm/s$'
                std = '$' + str(np.round(self.predict[k]['std_error'], 2)) + 'm/s$'

            ax[2, n].set_title('mean=' + mean + ' , std=' + std, fontsize=7)

            ax[2, n].hist(self.predict[k]['error'], bins=100, density=True,
                          facecolor='forestgreen', alpha=0.5, edgecolor=None, linewidth=0.5)

            ax[3, n].plot(self.model_data.history['loss'], '-', label='loss', color='blue', linewidth=1.0)
            ax[3, n].plot(self.model_data.history['val_loss'], '-', label='val_loss', color='red', linewidth=1.0)

        # cax = ax[1, 1].inset_axes([0.3, 1.0, 0.2, 0.04])
        # fig.colorbar(h[3], ax=ax[1, 1])

        cax = ax[1, 1].inset_axes([1.1, 0.0, 0.05, 1.0], label='# of points')
        cbar = fig.colorbar(h[3], cax=cax)
        cbar.set_label('# of points', rotation=270, fontsize=7, labelpad=8)
        cbar.ax.tick_params(labelsize=6)

        # Formatting
        for a in ax.flat:
            a.tick_params(axis='both', labelsize=6)

        for a in ax[0, :].flat:
            a.grid()

        for a in ax[0:2, :].flat:
            a.set_aspect(1.0)

        ax[3, -1].legend(bbox_to_anchor=(2.0, 1.0), fontsize=7)
        for a in ax[3, :]:
            a.set_xlabel('epochs', fontsize=7)
            a.set_ylabel('loss', fontsize=7)
            a.grid()
            a.set_yscale('log')

        if 'zeta' in self.output_names:
            for a in ax[0:2, :].flat:
                ff.pi_axis(a, axis_name='y')
                ff.pi_axis(a, axis_name='x')

                offset = 0.0
                a.set_ylim(-np.pi - offset, np.pi + offset)
                a.set_xlim(-np.pi - offset, np.pi + offset)

                a.set_ylabel('predicted $\zeta$ (rad)', fontsize=7)
                a.set_xlabel('true $\zeta$ (rad)', fontsize=7)

            for a in ax[2, :].flat:
                ff.pi_axis(a, axis_name='x')
                a.set_ylabel('density', fontsize=7)
                a.set_ylim(bottom=-0.05)
                a.set_xlabel('prediction error (rad)', fontsize=7)
                # a.grid()

        elif 'w' in self.output_names:
            for a in ax[0:1, :].flat:
                offset = 0.05
                a.set_ylim(0.0 - offset, 1.0 + offset)
                a.set_xlim(0.0 - offset, 1.0 + offset)

                a.set_ylabel('predicted $w$ (m/s)', fontsize=7)
                a.set_xlabel('true $w$ (m/s)', fontsize=7)

            for a in ax[2, :].flat:
                a.set_ylabel('density', fontsize=7)
                a.set_ylim(bottom=-0.05)
                a.set_xlabel('prediction error (m/s)', fontsize=7)
                # a.grid()

        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.9, hspace=0.6)


def segment_from_df(df, column='time', val=0.0):
    """ Pulls out segments from data frame based on where 'val' shows up in 'column'.
    """

    # Fins start of segments
    segment_start = np.where(df[column].values == val)[0].squeeze()  # where segments start
    n_segment = segment_start.shape[0]  # number of segments

    segment_list = []  # list of individual segments
    for n in range(n_segment):
        if n == (n_segment - 1):
            segment_end = df.shape[0]
        else:
            segment_end = segment_start[n + 1]

        segment = df.iloc[segment_start[n]:segment_end, :]
        segment_list.append(segment)

    return segment_list, n_segment
