import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
keras = tf.keras


class Model():

    def __init__(self, memory=1, p_step=1):
        self.memory = memory
        # The prediction step
        self.p_step = p_step

    def prepare_data(self, data, sample_index=[-1]):
        """Prepare data for training.

        :param data: The data to be prepared else the path to the data
        :type data: str/numpy.ndarray
        :param sample_index: The indices of the data points from data to prepare
            (default=[-1], this will select all data)
        :type sample_index: list(int)
        :returns: A tuple of numpy arrays representing the input and output data sets
        :rtype: tuple(numpy.ndarray)
        """
        # Load in the data if a path is specified
        if type(data) is str:
            with np.load(data) as infile:
                # If an index is provied, then select this data
                if sample_index[0] != -1:
                    self.data = infile['arr_0'][sample_index]
                else:
                    self.data = infile['arr_0']
        else:
            # else load all the data
            self.data = data

        #self.data = self.data[..., 1:3]

        # The size of the last axis
        self.vector_size = self.data.shape[-1]
        # This will be the size of the vector fed into the model
        self.input_size = self.memory*self.vector_size

        # The first element of x_arrays will store the values h timesteps in the
        # past, the next and h-1 time steps in the past and so on. The arrays are
        # then concatenated to give an array that contains a memory of h timesteps
        # prior to the currentyly being predicted one, which is the value in y
        x_arrays = []
        for i in range(self.memory):
            # This is the line that divides each sample individually into the
            # required feature vectors depending on the size of memory
            # This prevents us associating features from one sample with labels
            # of another sample
            #x_arrays.append(self.data[:, i:(-self.memory-self.p_step-1)+i, :])
            x_arrays.append(self.data[:, i:(-self.memory-self.p_step+1)+i, :])
        # This concatenate joins all features of lenght of 42 to form a single
        # feature vector of size 42*memory. Finally the reshape merges the data
        # from all samples together and removes the boundaries. It is important
        # that samples are dealt with individually before combining
        self.x = np.concatenate(x_arrays, axis=-1).reshape(-1,
                                                 self.memory*self.vector_size)
        self.y = self.data[:, self.memory+self.p_step-1:, :].reshape(-1, self.vector_size)

        return self.x, self.y

    def split(self, tr_size=-1):
        """Split the data into training and test data. Save the test data for later

        :param tr_size: The proportion of the data to use for training
            (default=-1, which will put all data into training set)
        :type tr_size: float
        :returns: A tuple of numpy arrays representing the training and test sets
        :rtype: tuple(numpy.ndarray)
        """
        if tr_size == -1:
            tr_size = self.x.shape[0] - 1



        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                            train_size=tr_size, random_state=42)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        return x_train, x_test, y_train, y_test

    def build_and_compile(self, lr, layers=None, loss=keras.losses.Huber()):
        """Build and compile the model.

        :param lr: The learning rate to use
        :type lr: float
        :returns: A compiled machine learning model
        :rtype: keras.model
        """
        # Define the model
        if layers is not None:
            model = keras.models.Sequential(layers)
        else:
            # A default configuration

            model = keras.models.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=[self.input_size]),
                #keras.layers.Dense(128, activation='relu'),
                #keras.layers.Dense(256, activation='relu'),
                #keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(self.vector_size)
            ])
            '''
            model = keras.models.Sequential([
                keras.layers.LSTM(64, input_shape=input_shape), #input_shape=[self.input_size]),
                #keras.layers.Dense(128, activation='relu'),
                #keras.layers.Dense(256, activation='relu'),
                #keras.layers.Dense(128, activation='relu'),
                #keras.layers.LSTM(64),
                keras.layers.Dense(self.vector_size)
            ])
            '''

        # Compile the model
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss=loss,
            optimizer=opt,
            metrics = ['mae']
        )
        self.model = model
        return model

    def fit(self, validation_split, out_file, batch_size=32, lr=1e-4,
            patience=5, epochs=1):
        """Fit the data to a model.

        :param validation_dataset: The dataset on which the model is to be validated
        :type validation_dataset: tensorflow.Dataset
        :param out_file: The name of the file in which to save the best model
        :type out_file: str
        :param batch_size: The batch size to use in training
        :type batch_size: int
        :param lr: The learning rate to use (default=1e-4)
        :type lr: float
        :param patience: The number of epoch to stop training after if there is no
            change in model performance
        :type patience: int
        :param epochs: The number of epochs to train for
        :type epochs: int
        :returns: The history of the model
        :rtype: keras.history
        """
        """Fit the finalised model."""
        early_stopping = keras.callbacks.EarlyStopping(patience=patience)
        checkpoint = keras.callbacks.ModelCheckpoint(out_file,
                                                    save_best_only = True)
        opt = keras.optimizers.Adam(learning_rate=lr)

        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 epochs=epochs,
                                 callbacks=[early_stopping, checkpoint])
        self.history = history
        return history

    @staticmethod
    def M_P_train(data_file, out_path, p_list, m_list, lr):
        """Train a collection of P models defined by the values in p_list."""

        for i, p_step in enumerate(p_list):
            for memory in m_list[i]:
                print(f'p_step = {p_step}')
                print(f'memory = {memory}')
                model = Model(memory=memory, p_step=p_step)
                model.prepare_data(data_file)
                model.split()

                model.build_and_compile(lr=lr)
                model.fit(
                          validation_split=0.1,
                          out_file=f'{out_path}/model_m{memory}_p{p_step}',
                          lr=lr
                          )
