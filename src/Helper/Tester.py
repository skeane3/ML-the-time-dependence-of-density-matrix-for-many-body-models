import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.linalg import logm
from .Model import Model
from .DataGenerator import DataGenerator

class Tester():

    @staticmethod
    def test_density_error(data, model_path, fig_path, p_list, n_data, t_tot, memory=1):
        """Calculate the predictions for each of the P models on the test data.
           This calculates the error on all samples in the test set.
           This predicts the value at P steps into the future using the exact
           values at each time step.
           Plot the resutls.


        :param data: The data to be prepared else the path to the data.
        :type data: str/numpy.ndarray
        :param model_path: The path to the P models.
        :type model_path: str
        :param fig_path: The path to save the figure.
        :type fig_path: str
        :param p_list: The list of prediction timesteps to be used
        :type p_list: list(int)
        :param n_data: The number of samples contained in actual and predictions
        :type n_data: int
        :param t_tot: The total time evolved for
        :type t_tot: int
        :param memory: The memory used for the models
        :type memory: int
        """

        # Load all our models into a dictionary, referenced by the P step they compute
        model_dict = {}
        for p_step in p_list:
            model_dict[p_step] = keras.models.load_model(f'{model_path}{p_step}')

        figs = []

        fig = plt.figure()
        fig.suptitle(r'Mean Element Error vs $k$')
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # Test every model
        for p_step, model in model_dict.items():
            # Create a DensityModel object for loading data
            density_model = Model(memory=memory, p_step=p_step)
            # Lead in the test data. Note we don't call split as this is only for training
            features, exact = density_model.prepare_data(data)
            # Calculate the predictios
            predictions = model.predict(features)

            diff = exact - predictions

            n_steps = diff.shape[0]//n_data
            vector_size = diff.shape[-1]

            diff = exact - predictions
            n_steps = diff.shape[0]//n_data
            time = np.linspace(0, t_tot, n_steps)

            vector_size = diff.shape[-1]
            diff = diff.reshape(n_data, n_steps, vector_size)
            # Computes sum over each 42-vector to get the cumulative error
            element_error = np.mean(np.abs(diff), axis=-1)
            # Compute the mean over the samples at each time step
            # Error should now contain the mean error at each time step
            error = np.mean(element_error, axis=0)
            # The average value of the error over all time steps
            mean_error = np.mean(error)

            ax.plot(time, error, linewidth=1, label=r'$k$ = %i' %(p_step))
            ax.set_ylim(0, 0.001)
            '''
            fig = plt.figure()
            fig.suptitle(f'Matrix Element Error P = {p_step}')
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            # Divide by 2 to separate the real and complex parts
            ax.scatter(np.linspace(0, t_tot, n_steps), error, s=0.1)
            ax.axhline(y=mean_error, xmin=0,xmax=t_tot, c='r',
                       label='Mean Error = %.4f' % mean_error)
            ax.set_ylabel('Time')
            ax.set_ylabel('Error')
            ax.legend()
            figs.append(fig)
            '''
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Mean Element Error')
        ax.legend()
        #figs.append(fig)
        '''
        with PdfPages(fig_path) as pp:
            for fig in figs:
                fig.savefig(pp, format='pdf', bbox_inches='tight')
        '''
        fig.savefig(fig_path, bbox_inches='tight')



    @staticmethod
    def memory_error(data, model_path, fig_path, p_list, m_list, n_data, t_tot, title):
        """Calculate the predictions for each of the P models on the test data.
           This calculates the error on all samples in the test set.
           This predicts the value at P steps into the future using the exact
           values at each time step.
           Plot the resutls.

        :param data: The data to be prepared else the path to the data.
        :type data: str/numpy.ndarray
        :param model_path: The path to the P models.
        :type model_path: str
        :param fig_path: The path to save the figure.
        :type fig_path: str
        :param p_list: The list of prediction timesteps to be used
        :type p_list: list(int)
        :param m_list: The list of memories to be used
        :type m_list: list(int)
        :param n_data: The number of samples contained in actual and predictions
        :type n_data: int
        :param t_tot: The total time evolved for
        :type t_tot: int
        """

        # Load all our models into a dictionary, referenced by the P step they compute
        model_dict = {}
        '''
        for p_step in p_list:
            for memory in m_list:
                model_dict[f'{memory}_{p_step}'] = keras.models.load_model(f'{model_path}_m{memory}_p{p_step}')
        '''
        #p = []
        #m = []
        for i, p_step in enumerate(p_list):
            for memory in m_list[i]:

                #p.append(p_step)
                #m.append(memory)
                model_dict[f'{memory}_{p_step}'] = keras.models.load_model(f'{model_path}_m{memory}_p{p_step}')
                #model = keras.models.load_model(f'{model_path}_m{memory}_p{p_step}')
                #config = model.get_config() # Returns pretty much every information about your model
                #print(f'P = {p_step}, m = {memory}, {config["layers"][0]["config"]["batch_input_shape"]}'
                #print(model.layers[-1].output_shape)

        errors = defaultdict(lambda: [[], []])

        figs = []
        # Test every model
        for key, model in model_dict.items():
            memory, p_step = key.split('_')
            # Create a DensityModel object for loading data
            density_model = Model(memory=int(memory), p_step=int(p_step))
            # Lead in the test data. Note we don't call split as this is only for training
            features, exact = density_model.prepare_data(data)
            # Calculate the predictios
            predictions = model.predict(features)

            diff = exact - predictions

            n_steps = diff.shape[0]//n_data
            vector_size = diff.shape[-1]

            diff = exact - predictions

            errors[p_step][0].append(memory)
            errors[p_step][1].append(np.mean(np.abs(diff)))

        fig = plt.figure(1)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ins = ax.inset_axes([0.6,0.6,0.35,0.35])
        min_errors = []
        # value[0] is memory
        # value[1] is error
        for key, value in errors.items():
            ax.plot(value[0], value[1], label=r'$k = %s$' %(key))
            min_errors.append(np.min(value[1]))
        ins.plot(errors.keys(), min_errors)
        ins.set_xlabel(r'$k$')
        ins.set_ylabel('Error')
        ax.set_xlabel('Memory')
        ax.set_ylabel('Error')
        fig.suptitle(title)
        ax.legend(loc='upper left')


        fig.savefig(fig_path, bbox_inches='tight')


    @staticmethod
    def test_density_autoreg(data, model_path, fig_path, p_list, evo_time, n_data):
        """Predicts the future value of the density matrix at each time step
           using an autoregression from some initial time. Plots the exact value
           of each matrix element vs the predicted values along time. i.e a
           trajectory of each element.

        :param data: The data to be prepared else the path to the data.
        :type data: str/numpy.ndarray
        :param model_path: The path to the P models.
        :type model_path: str
        :param fig_path: The path to save the figure.
        :type fig_path: str
        :param p_list: The list of prediction time steps to be used
        :type p_list: list(int)
        :param n_data: The number of individual trajectories in data
        :type n_data: int
        :param time_list: A list of times to perform the autoregression for
        :type time_list: list(int)
        :return: The predicted and exact valeus of the test data
        :rtype: tuple(list)
        """

        model_dict = {}
        for p_step in p_list:
            model_dict[p_step] = keras.models.load_model(f'{model_path}{p_step}')

        # Load in the test data
        # Although we are using different p_steps, we only actually need the
        # first one to make predictions on, and the one located at each time in
        # time_list to compare with. Hence we lead in the results with a p_step
        # of 1
        density_model = Model(memory=1, p_step=1)
        features, exact = density_model.prepare_data(data)
        features = features.reshape(n_data, -1, features.shape[-1])
        exact = exact.reshape(n_data, -1, exact.shape[-1])
        sample = np.random.randint(0, features.shape[0])
        features, exact = features[sample], exact[sample]

        # These values allow us to know how many time steps should be in each sample

        n_steps = exact.shape[0]
        vector_size = exact.shape[-1]//2

        prediction = features[0].reshape(1, -1)
        # prediction_times stores the times at which our predictions are made
        predict_times = []
        # predictions stores our predictions at each time in prediction_times
        predictions = []

        predict_times.append(0)
        predictions.append(prediction)

        # Evovle the state over the specified number of time steps with the
        # most appropriate model. For example, to predict time t = 371 we
        # would evolve by the p = 100 model 3 times, followed by the p = 50
        # model once, then the p = 20 model once and finally by the p = 1
        # model (assuming the models [1, 5, 10, 20, 50, 100])
        t = evo_time
        while t>0:
            for p_step in sorted(model_dict.keys(), reverse=True):
                # If the p_step value is less than the time left, then we
                # can use taht P_model to evolve
                if t//p_step >= 1:
                    prediction = model_dict[p_step].predict(prediction)
                    predictions.append(prediction)
                    predict_times.append(predict_times[-1] + p_step)
                    t -= p_step
                    break
        # Store the results for each sample at this time for plotting
        predictions = (np.array(predictions))
        predictions = predictions.reshape(-1, predictions.shape[-1])

        #act_times = exact[predict_times]

        predict_times = np.array(predict_times)
        time_array = np.arange(0, evo_time)


        rows = 2
        cols = 2

        fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False)
        fig.tight_layout(h_pad=3, w_pad=3)


        ax = axes[0, 0]
        ax.set_title(r'$\rho_{%i%i}$' %(0, 0))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(0, 0))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 0], linewidth=0.5)
        for j in range(len(predictions)):
                ax.scatter(predict_times[j], predictions[j][0], s=5, c='red')

        ax = axes[0, 1]
        ax.set_title(r'$\rho_{%i%i}$' %(1, 3))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(1, 3))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 8], linewidth=0.5)
        for j in range(len(predictions)):
                ax.scatter(predict_times[j], predictions[j][8], s=5, c='red')

        ax = axes[1, 0]
        ax.set_title(r'$\rho_{%i%i}$' %(2, 4))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(2, 4))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 13], linewidth=0.5)
        for j in range(len(predictions)):
                ax.scatter(predict_times[j], predictions[j][13], s=5, c='red')

        ax = axes[1, 1]
        ax.set_title(r'$\rho_{%i%i}$' %(4, 5))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(4, 5))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 19], linewidth=0.5)
        for j in range(len(predictions)):
                ax.scatter(predict_times[j], predictions[j][19], s=5, c='red')


        fig.savefig(fig_path, bbox_inches='tight')

    @staticmethod
    def test_occupation_autoreg(data, model_path, fig_path, p_list, evo_time, n_data, memory):
        """Predicts the future value of the occupation at each time step
           using an autoregression from some initial time. Plots the exact value
           of each occupation vs the predicted values along time. i.e a
           trajectory of each element.

        :param data: The data to be prepared else the path to the data.
        :type data: str/numpy.ndarray
        :param model_path: The path to the P models.
        :type model_path: str
        :param fig_path: The path to save the figure.
        :type fig_path: str
        :param p_list: The list of prediction time steps to be used
        :type p_list: list(int)
        :param n_data: The number of individual trajectories in data
        :type n_data: int
        :param time_list: A list of times to perform the autoregression for
        :type time_list: list(int)
        :return: The predicted and exact valeus of the test data
        :rtype: tuple(list)
        """


        model = keras.models.load_model(model_path)
        # Load in the test data
        # Although we are using different p_steps, we only actually need the
        # first one to make predictions on, and the one located at each time in
        # time_list to compare with. Hence we lead in the results with a p_step
        # of 1
        density_model = Model(memory=memory, p_step=1)
        features, exact = density_model.prepare_data(data)
        features = features.reshape(n_data, -1, features.shape[-1])
        exact = exact.reshape(n_data, -1, exact.shape[-1])
        sample = 128#np.random.randint(0, features.shape[0])
        features, exact = features[sample], exact[sample]

        # These values allow us to know how many time steps should be in each sample

        n_steps = exact.shape[0]
        vector_size = exact.shape[-1]


        # prediction_times stores the times at which our predictions are made
        predict_times = np.arange(1, evo_time+1)
        # predictions stores our predictions at each time in prediction_times
        predictions = np.empty(((memory+evo_time), vector_size))
        predictions[:memory] = features[0].reshape(memory, -1)


        # Evovle the state over the specified number of time steps with the
        # most appropriate model. For example, to predict time t = 371 we
        # would evolve by the p = 100 model 3 times, followed by the p = 50
        # model once, then the p = 20 model once and finally by the p = 1
        # model (assuming the models [1, 5, 10, 20, 50, 100])
        t = evo_time
        i = memory

        while t>0:
            #print(predictions)
            '''
            prediction_temp = model_dict['m2_p1'].predict(prediction.reshape(1, -1))
            predictions.append(prediction_temp.copy())
            prediction[0] = prediction[-1]
            prediction[-1] = prediction_temp
            predict_times.append(predict_times[-1] + 1)
            '''
            temp = predictions[i-memory:i].reshape(1, -1)
            print(f'temp = {temp}')
            predict = model.predict(temp)[0]
            print(f'predict={predict}')
            predictions[i] = predict
            i += 1
            t -= 1


        # Store the results for each sample at this time for plotting
        predictions = (np.array(predictions))
        predictions = predictions.reshape(-1, predictions.shape[-1])
        print(predictions.shape)
        print(predictions)
        print(len(predict_times))


        nrows = 2
        ncols = 2
        time_array = np.arange(1, evo_time+1)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        fig.suptitle(f'Occupation Trajectory Memory')
        fig.tight_layout(h_pad=2, w_pad=2)
        for i in range(nrows*ncols):
            ax = axes[i//nrows, i%ncols]
            ax.set_title(r'$n_{%i}$' %(i+1))
            ax.plot(time_array, exact[memory:evo_time+memory, i], '-.', linewidth=0.5, label='Exact')
            ax.scatter(predict_times, predictions[memory:, i], s=2, c='orange', label='Prediction')
            #ax.plot(time_array, np.abs(exact[memory:evo_time+memory, i]-predictions[memory:, i]), '-.', linewidth=0.5, label='Difference')
            ax.set_xlabel('Prediction Steps')
            ax.set_ylabel(r'$n_{%i}$' %(i+1))
            ax.set_ylim(0, 1)
            ax.legend()


        fig.savefig(fig_path, bbox_inches='tight')


    @staticmethod
    def test_entropy(density_data, occupation_data, fig_path, n_data, t_tot, n_samples=1):
        """Plot the entropy of n_samples samples from the test data set."""

        model = Model(memory=1, p_step=1)
        _, exact_density = model.prepare_data(density_data)
        _, exact_occupation = model.prepare_data(occupation_data)

        exact_density = exact_density.reshape(n_data, -1, exact_density.shape[-1])
        exact_occupation = exact_occupation.reshape(n_data, -1, exact_occupation.shape[-1])

        sample_indices = np.random.randint(low=0, high=exact_density.shape[0], size=n_samples)

        density_samples = DataGenerator.vector_to_matrix(exact_density[sample_indices])
        occupation_samples = exact_occupation[sample_indices]

        n_steps = exact_density.shape[1]

        occupation_entropy = -1*np.sum(occupation_samples*np.log(occupation_samples), axis=-1)

        time = np.linspace(0, t_tot, n_steps)

        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
        ax.set_title(r'Entropy $(S)$')

        times = [[0, 10], [700, 710], [950, 960]]
        for time in times:
            ax.hist(occupation_entropy[:, time[0]:time[1]].flatten(), density=True, bins=220, histtype='step', label=f't = {int(20*time[0]/1000)+1}')#, bins=np.linspace(0, np.max(entropy), 2))

        ax.set_xlabel(r'$S$')
        ax.set_ylabel('Frequency')
        ax.legend()

        fig.savefig(fig_path, bbox_inches='tight')



    def test_density_n(model_path, fig_path, p_list, n_evo, data_generator, n_save, memory=1):
        """This methods plots how the error of each model accumulates with each
           application of the autoregression. The idea is to see how many times
           our models can be applied before the accumulated error becomes too
           large to give valid results.
        """

        model_dict = {}
        for p_step in p_list:
            model_dict[p_step] = keras.models.load_model(f'{model_path}{p_step}')

        n_data=1
        density = data_generator.get_initial_densities(n_data=n_data)
        density = data_generator.evolve_density(density, n_save=n_save)
        density = DataGenerator.matrix_to_vector(density)

        figs = []
        fig = plt.figure()
        fig.suptitle(f'Autoregression Error')
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        time = np.arange(0, n_evo)

        # Test every model
        for p_step, model in model_dict.items():
            # Create a DensityModel object for loading data
            density_model = Model(memory=memory, p_step=p_step)
            # Lead in the test data. Note we don't call split as this is only for training
            features, exact = density_model.prepare_data(density)
            # Calculate the predictios

            features = features.reshape(n_data, -1, features.shape[-1])
            exact = exact.reshape(n_data, -1, exact.shape[-1])

            predict_times = np.arange(0, n_evo*p_step, p_step)
            exact = exact[:, predict_times]
            predictions = np.empty((n_data, n_evo, features.shape[-1]))


            predictions[:, 0] = model.predict(features[:, 0].reshape(features.shape[0], -1))


            for i in range(1, n_evo):
                predictions[:, i] = model.predict(predictions[:, i-1].reshape(predictions.shape[0], -1))

            diff = exact - predictions

            n_steps = diff.shape[0]//n_data
            vector_size = diff.shape[-1]

            diff = exact - predictions

            vector_size = diff.shape[-1]

            diff = diff.reshape(n_data, n_evo, vector_size)
            # Computes sum over each 42-vector to get the cumulative error
            element_error = np.mean(np.abs(diff), axis=-1)
            # Compute the mean over the samples at each time step
            # Error should now contain the mean error at each time step
            error = np.mean(element_error, axis=0)

            ax.plot(time, error, label=r'$k = %s$' %(p_step))

        ax.set_xlabel('Prediction Steps')
        ax.set_ylabel('Error')
        ax.legend()

        fig.savefig(fig_path, bbox_inches='tight')

    def test_occupation_n(model_path, fig_path, p, model_list, memory_list, n_evo, data_generator, n_save, memory=1):
        """This methods plots how the error of each model accumulates with each
           application of the autoregression. The idea is to see how many times
           our models can be applied before the accumulated error becomes too
           large to give valid results.
        """

        model_dict = {}
        for model_name in model_list:
            model_dict[model_name] = keras.models.load_model(f'{model_path}/{model_name}')

        n_data=1
        density = data_generator.get_initial_densities(n_data=n_data)
        occupation = data_generator.evolve_occupation(density, n_save=n_save)

        figs = []
        fig = plt.figure()
        fig.suptitle(f'Autoregression Error')
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        time = np.arange(0, n_evo)

        # Test every model
        for memory, model in zip(memory_list, model_dict.values()):
            # Create a DensityModel object for loading data
            occ_model = Model(memory=memory, p_step=p)
            # Lead in the test data. Note we don't call split as this is only for training
            features, exact = occ_model.prepare_data(occupation)
            # Calculate the predictios


            features = features.reshape(n_data, -1, features.shape[-1])
            exact = exact.reshape(n_data, -1, exact.shape[-1])

            #print(features[:, 1:] == exact[:, :-1])

            predict_times = np.arange(0, n_evo*p_step, p_step)
            exact = exact[:, predict_times]
            predictions = np.empty((n_data, n_evo, features.shape[-1]))


            predictions[:, 0] = model.predict(features[:, 0].reshape(features.shape[0], -1))


            for i in range(1, n_evo):
                predictions[:, i] = model.predict(predictions[:, i-1].reshape(predictions.shape[0], -1))

            diff = exact - predictions

            n_steps = diff.shape[0]//n_data
            vector_size = diff.shape[-1]

            diff = exact - predictions

            vector_size = diff.shape[-1]

            diff = diff.reshape(n_data, n_evo, vector_size)
            # Computes sum over each 42-vector to get the cumulative error
            element_error = np.mean(np.abs(diff), axis=-1)
            # Compute the mean over the samples at each time step
            # Error should now contain the mean error at each time step
            error = np.mean(element_error, axis=0)

            ax.plot(time, error, label=r'$k = %s$' %(p_step))

        ax.set_xlabel('Prediction Steps')
        ax.set_ylabel('Error')
        ax.legend()

        fig.savefig(fig_path, bbox_inches='tight')



    @staticmethod
    def test_density_autoreg_all(data, model_path, fig_path, p_list, evo_time, n_data):
        """Predicts the future value of the density matrix at each time step
           using an autoregression from some initial time. Plots the exact value
           of each matrix element vs the predicted values along time. i.e a
           trajectory of each element.

        :param data: The data to be prepared else the path to the data.
        :type data: str/numpy.ndarray
        :param model_path: The path to the P models.
        :type model_path: str
        :param fig_path: The path to save the figure.
        :type fig_path: str
        :param p_list: The list of prediction time steps to be used
        :type p_list: list(int)
        :param n_data: The number of individual trajectories in data
        :type n_data: int
        :param time_list: A list of times to perform the autoregression for
        :type time_list: list(int)
        :return: The predicted and exact valeus of the test data
        :rtype: tuple(list)
        """

        model_dict = {}
        for p_step in p_list:
            model_dict[p_step] = keras.models.load_model(f'{model_path}{p_step}')

        # Load in the test data
        # Although we are using different p_steps, we only actually need the
        # first one to make predictions on, and the one located at each time in
        # time_list to compare with. Hence we lead in the results with a p_step
        # of 1
        density_model = Model(memory=1, p_step=1)
        features, exact = density_model.prepare_data(data)
        features = features.reshape(n_data, -1, features.shape[-1])
        exact = exact.reshape(n_data, -1, exact.shape[-1])
        sample = np.random.randint(0, features.shape[0])
        features, exact = features[sample], exact[sample]

        # These values allow us to know how many time steps should be in each sample

        n_steps = exact.shape[0]
        vector_size = exact.shape[-1]//2

        initial_state = features[0].reshape(1, -1)
        # prediction_times stores the times at which our predictions are made

        # predictions stores our predictions at each time in prediction_times
        predictions = np.empty_like(exact)
        predictions[0] = initial_state

        # Evovle the state over the specified number of time steps with the
        # most appropriate model. For example, to predict time t = 371 we
        # would evolve by the p = 100 model 3 times, followed by the p = 50
        # model once, then the p = 20 model once and finally by the p = 1
        # model (assuming the models [1, 5, 10, 20, 50, 100])
        for t in range(1, evo_time):
            for p_step in sorted(model_dict.keys(), reverse=True):
                # If the p_step value is less than the time left, then we
                # can use taht P_model to evolve
                if t%p_step == 0:
                    predictions[t] = model_dict[p_step].predict(predictions[t-p_step].reshape(1, -1))

        # Store the results for each sample at this time for plotting

        #act_times = exact[predict_times]

        time_array = np.arange(0, evo_time)

        rows = 2
        cols = 2

        fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False)
        fig.tight_layout(h_pad=3, w_pad=3)
        print(predictions.shape)

        ax = axes[0, 0]
        ax.set_title(r'$\rho_{%i%i}$' %(0, 0))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(0, 0))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 0], linewidth=0.5, label='Exact')
        ax.plot(time_array, predictions[:evo_time, 0], '-.', linewidth=0.5, c='red', label='Prediction')

        ax = axes[0, 1]
        ax.set_title(r'$\rho_{%i%i}$' %(1, 3))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(1, 3))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 8], linewidth=0.5, label='Exact')
        ax.plot(time_array, predictions[:evo_time, 8], '-.', linewidth=0.5, c='red', label='Prediction')

        ax = axes[1, 0]
        ax.set_title(r'$\rho_{%i%i}$' %(2, 4))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(2, 4))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 13], linewidth=0.5, label='Exact')
        ax.plot(time_array, predictions[:evo_time, 13], '-.', linewidth=0.5, c='red', label='Prediction')

        ax = axes[1, 1]
        ax.set_title(r'$\rho_{%i%i}$' %(4, 5))
        ax.set_ylabel(r'$\rho_{%i%i}$' %(4, 5))
        ax.set_xlabel('Time')
        ax.plot(time_array, exact[:evo_time, 19], linewidth=0.5, label='Exact')
        ax.plot(time_array, predictions[:evo_time, 19], '-.', linewidth=0.5, c='red', label='Prediction')


        fig.savefig(fig_path, bbox_inches='tight')
