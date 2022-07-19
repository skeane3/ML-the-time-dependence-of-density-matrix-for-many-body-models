import numpy as np
from scipy.linalg import logm


class DataGenerator():


    def __init__(self, system, n_data, t_tot, dt):

        self.system = system
        self.n_data = n_data
        self.t_tot = t_tot
        self.dt = dt
        self.n_steps = int(t_tot/dt)

    def runge_kutta(self, rho, H):
        """ Calculate the evolution of the density matrix using RK4 method.

        :param rho: The density matrix at time t
        :type rho: numpy.ndarray
        :param H: The Hamiltonian that the system evolves under
        :type H: numpy.ndarray
        :return: The amount to update rho by to increment it to time t + self.dt
        :rtype: numpy.ndarray
        """
        # Calculate the 4 Runge Kutta coefficients
        def commutator(A, B):
            """Compute the commuttor of two matrices.

            :param A: A matrix
            :type A: numpy.ndarray
            :param B: A matrix
            :type B: numpy.ndarray
            :return: The commutator of A nd B
            :rtype: numpy.ndarray
            """
            return A@B - B@A

        k1 = -1j*commutator(H, rho)
        k2 = -1j*commutator(H, rho + 0.5*self.dt*k1)
        k3 = -1j*commutator(H, rho + 0.5*self.dt*k2)
        k4 = -1j*commutator(H, rho + self.dt*k3)

        return (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)

    def evolve_density(self, initial_densities, n_save=-1):
        """Evolve multiple states in time simultaneously.

        :param initial_densities: An collection of initial states to evolve. This
            should be of shape (n_data, n_steps, density_matrix_size,
            density_matrix_size) where density_matrix_size is the number of rows
            (or colmns as its a square matrix)
        :type initial_densities: numpy.ndarray
        :param n_save: The number of timesteps to return
        :type n_save: int
        :return: The time evolution of the density matrix of each sample
            shape (n_data, n_steps, density_matrix_size,
                                density_matrix_size)
        :rtype: numpy.ndarray
        """

        # If timesteps to save is not given, then save all
        if n_save == -1:
            n_save = self.n_steps

        # We will save the state every save_step number of timesteps
        save_step = int(self.n_steps/n_save)

        n_initial_densities = initial_densities.shape[0]
        # Define an array to store the density matrix at each time
        # The first axis represents trajectories of individual states and the
        # second is the time evolution of each distint state
        evolution_array = np.empty((n_initial_densities, n_save,
                                    self.system.n_states, self.system.n_states),
                                    dtype=complex)

        # Set the first column of the evolution to the initial states
        # (at time t = 0)
        density = initial_densities
        evolution_array[:, 0, ...] = initial_densities
        # Evolve each state in time independently
        for i in range(1, self.n_steps):
            # Update the density matrix in time
            density += self.runge_kutta(density, self.system.H)
            # If we have evolved save_step number of steps, then store the state
            if i%save_step == 0:
                evolution_array[:, i//save_step, ...] = density

        return evolution_array

    def evolve_occupation(self, initial_densities, n_save=-1):
        """Evolve multiple states in time simultaneously, but return the
           occupation at each time step, not the density matrix.

        :param initial_densities: An collection of initial states to evolve. This
            should be of shape (n_data, n_steps, density_matrix_size,
            density_matrix_size) where density_matrix_size is the number of rows
            (or colmns as its a square matrix)
        :type initial_densities: numpy.ndarray
        :param n_save: The number of timesteps to return
        :type n_save: int
        :return: The time evolution of the site occupation of each sample
            shape (n_data, n_steps, L)
        :rtype: numpy.ndarray
        """
        n_initial_densities = initial_densities.shape[0]
        # If timesteps to save is not given, then save all
        if n_save == -1:
            n_save = self.n_steps
        # We will save the state at every save_step number of timesteps
        save_step = int(self.n_steps/n_save)

        # Define an array to store the occupation of each site at each time
        # The first axis represents trajectories of individual states, the
        # second represents individual sites and the third is the time axis
        occupation_array = np.empty((n_initial_densities, n_save, self.system.L),
                                    dtype=np.float64)

        # Set the first column of the evolution to the initial states
        # (at time t = 0)
        density = initial_densities
        for j, op in enumerate(self.system.number_operators):
            occupation_array[:, 0, j] = np.real(np.trace(np.matmul(op, density),
                                                         axis1=-2, axis2=-1))

        # Evolve each state in time independently
        for i in range(1, self.n_steps):
            # Evolve the density matrix in time by one step
            density += self.runge_kutta(density, self.system.H)
            if i%save_step == 0:
                # Compute the occupations at this new timestep, if it is after
                # save_step number of timesteps
                for j, op in enumerate(self.system.number_operators):
                    occupation_array[:, i//save_step, j] = np.real(np.trace(np.matmul(op, density), axis1=-2, axis2=-1))

        return occupation_array

    def evolve_all(self, initial_densities, n_save=-1):
        """Evolve multiple states in time simultaneously, and return occupation
           and density matrix

        :param initial_densities: An collection of initial states to evolve. This
            should be of shape (n_data, n_steps, density_matrix_size)
        :type initial_densities: numpy.ndarray
        :param n_save: The number of timesteps to return
        :type n_save: int
        :return: The time evolution of the density matrix of each sample
            shape (n_data, n_steps, density_matrix_size, density_matrix_size)
            and the time evolution of the site occupation of each sample
            shape (n_data, n_steps, L)
        :rtype: numpy.ndarray
        """
        n_initial_densities = initial_densities.shape[0]
        if n_save == -1:
            n_save = self.n_steps
        # We will save the state at every save_step number of timesteps
        save_step = int(self.n_steps/n_save)

        # Define an array to store the densitu matrix at each time
        evolution_array = np.empty((n_initial_densities, n_save,
                                    self.system.n_states, self.system.n_states),
                                    dtype=complex)
        # Define an array to store the occupation of each site at each time
        occupation_array = np.empty((n_initial_densities, n_save, self.system.L),
                                    dtype=np.float64)

        # Set the first column of the evolution to the initial states (at time t = 0)
        density = initial_densities
        evolution_array[:, 0, ...] = initial_densities
        for j, op in enumerate(self.system.number_operators):
            occupation_array[:, 0, j] = np.real(np.trace(np.matmul(op, density), axis1=-2, axis2=-1))

        # Evolve each state in time independently
        for i in range(1, self.n_steps):
            # Evolve the density matrix in time by one step
            density += self.runge_kutta(density, self.system.H)
            if i%save_step == 0:
                evolution_array[:, i//save_step, ...] = density

                # Compute the occupations at this new timestep
                for j, op in enumerate(self.system.number_operators):
                    occupation_array[:, i//save_step, j] = np.real(np.trace(np.matmul(op, density), axis1=-2, axis2=-1))

        return evolution_array, occupation_array


    def get_initial_densities(self, n_data=None):
        """Generate n_data initaial states.

        :param n_data: The number of initial states to generate (default=None)
        :type n_data: int
        :return: Initial density matrices of shape (n_data, density_matrix_size,
            density_matrix_size)
        :rtype: numpy.ndarray"""
        # If a value of n_data is passed, then override the current one
        if n_data is not None:
            self.n_data = n_data
        # Generate n_data random arrays, with one coefficient for each basis state
        # Note that this is from a normal distribution, which when normalised
        # will be uniformly distributed on an n-sphere, where n is the number
        # of basis states (which is system.n_states)
        random_coefficients =  np.random.normal(0, 1,
                                            size=(self.n_data, self.system.n_states))
        # Get the norm of each random state
        norm_array = np.sqrt(np.sum(random_coefficients**2, axis=-1)).reshape(-1, 1)
        # Normalise each state
        random_states = random_coefficients/norm_array
        # Compute the density matrix representation of each random state vector
        initial_densities = np.empty((self.n_data, self.system.n_states, self.system.n_states),
                                    dtype=complex)
        for i, state in enumerate(random_states):
            initial_densities[i] = np.outer(state, state.conjugate())

        return initial_densities

    @staticmethod
    def matrix_to_vector(matrix):
        """Convert a collection square matriices to a vectors of their upper
        diaogonal components.

        :param matrix: A numpy array of matrices. This should be of shape
            (n_data, n_steps, density_matrix_size, density_matrix_size)
        :type matrix: numpy.ndarray
        :returns: A collection of vectors of upper diaogonal components.
        :rtype: numpy.ndarray
        """
        ########################################################################
        ###               WITH separation of complex components              ###
        ########################################################################

        # This refers to the number of rows/columns of the matrix. eg if matrix
        # is 6 x 6, then matrix_size will be 6, NOT 36
        matrix_size = matrix.shape[-1]
        # This refers to the number of components on and above the diagonal of
        # matrix, which will be stored in a single array of size vector size
        vector_size = int(matrix_size*(matrix_size+1)/2)
        # Define arrays to extract these upper triangular components
        index_array_1, index_array_2 = np.triu_indices(matrix_size)
        vector = matrix[..., index_array_1, index_array_2]
        # Split into real and complex parts. NOTE: This will make each element
        # on the last axis full_vector of length 2*vector_size
        full_vector = np.concatenate((vector.real, vector.imag), axis=-1).astype(np.float32)      # Separate real and complex values         ##############################################
        return full_vector

        ########################################################################
        ###             WITHOUT separation of complex components             ###
        ########################################################################
        '''
        # This refers to the number of rows/columns of the matrix. eg if matrix
        # is 6 x 6, then matrix_size will be 6, NOT 36
        matrix_size = matrix.shape[-1]
        # This refers to the number of components on and above the diagonal of
        # matrix, which will be stored in a single array of size vector size
        vector_size = int(matrix_size*(matrix_size+1)/2)
        # Define arrays to extract these upper triangular components
        index_array_1, index_array_2 = np.triu_indices(matrix_size)
        vector = matrix[..., index_array_1, index_array_2]
        return vector
        '''

    @staticmethod
    def vector_to_matrix(vector):
        """Convert a collection of arrays containing upper diaogonal components
           to a square matrix.

        :param vector: A vector of upper triangular components of shape
            (n_data, n_steps, vector_size)
        :type vector: numpy.ndarray
        :returns: A symmetric matrix built out of vector's components
        :rtype: numpy.ndarray
        """
        ########################################################################
        ###               WITH separation of complex components              ###
        ########################################################################
        vector_size = vector.shape[-1]//2                                                                # divide by two as our vector will contain real anc complex comonents separately ##################
        # matrix_size refers to the number of rows/columns of the resulting
        # matrix. The size of the resulting matrix will be
        # matrix_size*matrix_size
        matrix_size = int((-1 + np.sqrt(1+8*vector_size))/2)
        matrix = np.zeros(vector.shape[0]*vector.shape[1]*matrix_size*matrix_size,
                  dtype=complex).reshape(vector.shape[0], vector.shape[1], matrix_size, matrix_size)
        # Indices used to extract values from the vector and place into matrix
        lower_index = 0
        upper_index = matrix_size
        for i in range(matrix_size):
            matrix[..., i, i:] = vector[..., lower_index:upper_index] + 1j*vector[..., vector_size + lower_index:vector_size + upper_index]        # The second term adds in the complex values
            lower_index = upper_index
            upper_index += matrix_size - (i+1)
        conj = matrix.transpose(0, 1, 3, 2).conjugate()
        for i in range(matrix_size):
            conj[..., i, i] = 0
        # Fill in the lower triangular components
        return matrix + conj

        ########################################################################
        ###             WITHOUT separation of complex components             ###
        ########################################################################
        '''
        vector_size = len(vector)
        # The size of the resulting matrix will be size*size
        matrix_size = int((-1 + np.sqrt(1+8*vector_size))/2)
        matrix = np.zeros(matrix_size*matrix_size, dtype=complex).reshape(matrix_size, matrix_size)

        lower_index = 0
        upper_index = matrix_size
        for i in range(matrix_size):
            matrix[i, i:] = vector[lower_index:upper_index]
            lower_index = upper_index
            upper_index += matrix_size - (i+1)
        # Fill in the lower triangular components
        matrix = matrix + matrix.T.conjugate() - matrix.diagonal()
        return matrix
        '''
    def density_to_occupation(self, density_array):
        """Calculate the occupation of each site at each time from the density
           matrix information.

        :param matrix: A numpy array of the density matrices in their VECTOR
            FORM. This should be of shape (n_data, n_steps, vector_size)
        :type matrix: numpy.ndarray
        :returns: The time evolution of the site occupation of each sample
            shape (n_data, n_steps, L)
        :rtype: numpy.ndarray"""
        # First, convert back to matrix form
        density_matrix = DataGenerator.vector_to_matrix(density_array)
        occupation_array = np.empty((density_matrix.shape[0],
                                     density_matrix.shape[1], self.system.L),
                                     dtype=np.float64)

        # Set the first column of the evolution to the initial states (at time t = 0)
        for j, op in enumerate(self.system.number_operators):
            occupation_array[:, 0, j] = np.real(np.trace(np.matmul(op, density_matrix[:, 0]), axis1=-2, axis2=-1))

        # Evolve each state in time independently
        for i in range(1, density_matrix.shape[1]):
            # Compute the occupations at this new timestep
            for j, op in enumerate(self.system.number_operators):
                occupation_array[:, i, j] = np.real(np.trace(np.matmul(op, density_matrix[:, i]), axis1=-2, axis2=-1))

        return occupation_array

    def save_density(self, data, file_path):
        """Vectorise the data and save to output file.

        :param data: The data to be saved of shape (n_data, n_steps,
            density_matrix_size, density_matrix_size)
        :type data: numpy.ndarray
        :param file_path: The path to the file where the data is to be stored
        :type file_path: str
        """
        # First convert the data to vector representation.
        vectorised_data = self.matrix_to_vector(data)
        # If n_save not specified, then save all the data

        # Save the vectorised data in a compressed format
        np.savez_compressed(file_path, vectorised_data)

    def generate_density(self, file_path, n_save=-1):
        """Compute and save the the time evolution of density matrix data.

        :param data_file: The path to the file where the data is to be stored
        :type data_file: str
        :param n_save: The number of timesteps to save (default=-1)
        :type n_save: int
        """
        initial_densities = self.get_initial_densities()
        # Evolve the initial_densities with the given propagator according to
        # the specified system
        evolution_densities = self.evolve_density(initial_densities, n_save=n_save)
        self.save_density(data=evolution_densities, file_path=file_path)

    def save_occupation(self, data, file_path):
        """Vectorise the data and save to output file.

        :param data: The data to be saved
        :type data: numpy.ndarray
        :param file_path: The path to the file where the data is to be stored
        :type file_path: str
        """

        # Save the vectorised data in a compressed format
        np.savez_compressed(file_path, data)


    def generate_occupation(self, file_path, n_save=-1):
        """Compute and save the the time evolution of site occupation data.

        :param file_path: The path to the file where the data is to be stored
        :type file_path: str
        :param n_save: The number of timesteps to save (default=-1)
        :type n_save: int
        """
        initial_densities = self.get_initial_densities()
        # Evolve the initial_densities with the given propagator according to
        # the specified system
        evolution_occupation = self.evolve_occupation(initial_densities, n_save=n_save)

        self.save_occupation(data=evolution_occupation, file_path=file_path)

    def generate_data(self, density_path, occupation_path, n_save=-1):
        """Compute and save the the time evolution of density matrix  and site
           occupation data.

        :param density_path: The path to the file where the density data is to
            be stored
        :type file_path: str
        :param occupation_path: The path to the file where the occupation data
            is to be stored
        :type file_path: str
        :param n_save: The number of timesteps to save (default=-1)
        :type n_save: int
        """

        initial_densities = self.get_initial_densities()
        #print(initial_densities.shape)

        #initial_state = np.array([1, 0, 0, 0, 0, 0], dtype=complex)
        #initial_densities = np.outer(initial_state, initial_state)
        #initial_densities = initial_densities[np.newaxis, ...]

        # Evolve the initial_densities with the given propagator according to
        # the specified system
        evolution_densities, evolution_occupation = self.evolve_all(
                                                        initial_densities,
                                                        n_save=n_save)
        self.save_density(data=evolution_densities, file_path=density_path)
        self.save_occupation(data=evolution_occupation,
                             file_path=occupation_path)

    '''
    def random_samples(self, n_samples, n_save):
        initial_densities = self.get_initial_densities(n_data=n_samples)
        data = self.evolve_density(initial_states=initial_densities,
            n_save=n_save)
        return self.matrix_to_vector(data)
    '''

    def conserved_quantities(self, fig_path, dt_list):
        """Evolve multiple states in time simultaneously, and return occupation
           and density matrix

        :param initial_densities: An collection of initial states to evolve. This
            should be of shape (n_data, n_steps, density_matrix_size)
        :type initial_densities: numpy.ndarray
        :param n_save: The number of timesteps to return
        :type n_save: int
        :return: The time evolution of the density matrix of each sample
            shape (n_data, n_steps, density_matrix_size, density_matrix_size)
            and the time evolution of the site occupation of each sample
            shape (n_data, n_steps, L)
        :rtype: numpy.ndarray
        """
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
        ax.set_title('Conserved Quantities')
        ax.set_ylabel('Conserved Values')
        ax.set_xlabel('Time')
        initial_densities = self.get_initial_densities(n_data = 1)
        n_initial_densities = initial_densities.shape[0]

        for dt in dt_list:
            self.dt = dt


            # Define an array to store the densitu matrix at each time
            evolution_array = np.empty((n_initial_densities, self.n_steps,
                                        self.system.n_states, self.system.n_states),
                                        dtype=complex)
            # Define an array to store the occupation of each site at each time
            occupation_array = np.empty((n_initial_densities, self.n_steps, self.system.L),
                                        dtype=np.float64)

            # Set the first column of the evolution to the initial states (at time t = 0)
            density = initial_densities
            evolution_array[:, 0, ...] = initial_densities
            for j, op in enumerate(self.system.number_operators):
                occupation_array[:, 0, j] = np.real(np.trace(np.matmul(op, density), axis1=-2, axis2=-1))

            # Evolve each state in time independently
            for i in range(1, self.n_steps):
                # Evolve the density matrix in time by one step
                density += self.runge_kutta(density, self.system.H)
                evolution_array[:, i, ...] = density

                # Compute the occupations at this new timestep
                for j, op in enumerate(self.system.number_operators):
                    occupation_array[:, i, j] = np.real(np.trace(np.matmul(op, density), axis1=-2, axis2=-1))

            occupation_sum = np.sum(occupation_array, axis=-1).flatten()
            energy = np.real(np.trace(np.matmul(self.system.H, evolution_array), axis1=-2, axis2=-1)).flatten()
            trace = np.real(np.trace(evolution_array, axis1=-2, axis2=-1)).flatten()
            entropy = np.empty((evolution_array.shape[0], evolution_array.shape[1], 1), dtype=np.float32)
            for i in range(evolution_array.shape[0]):
                print(i)
                for j in range(evolution_array.shape[1]):
                    print(j)
                    entropy[i, j] = -1*np.real(np.trace(np.matmul(evolution_array[i, j], logm(evolution_array[i, j]))))

            time = np.linspace(0, self.t_tot, self.n_steps)

            ax.plot(time, occupation_sum, label='Particle Number')#label=r'$\sum_i Tr(\hat{n}_i\hat{\rho}(t))$')
            ax.plot(time, energy, label='Energy')#label = r'$Tr(\hat{H}\hat{\rho}(t))$')
            ax.plot(time, trace, label=r'Trace$(\hat{\rho})$')#label=r'$Tr(\hat{\rho}(t))$')
            ax.plot(time, entropy.flatten(), label='Entropy')#label=r'$-Tr(\hat{\rho}(t)ln(\hat{\rho}(t)))$')

        ax.legend()

        fig.savefig(fig_path, bbox_inches='tight')
        #print(occupation_sum)
        #print(energy)
        #print(trace)
        #return occupation_sum, , occupation_array
