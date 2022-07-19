import numpy as np
from scipy import linalg
from itertools import permutations



class Hubbard():

    def __init__(self, L, ne, t=1, V=0, E=[0, 0, 0, 0]):
        self.L = L          # The number of sites/atoms
        self.ne = ne        # The number of electrons
        self.t = t          # The energy scale of the system
        self.V = V          # The external potential of the problem
        self.E = E          # A list of onsite energies of each atom

        # A dictionary mapping the braket representation to vector representation
        self.basis_map = self.get_basis_map()
        self.n_states = len(self.basis_map)        # The number of basis states
        # Extract the states and basis vectors into lists
        self.states = list(self.basis_map.keys())
        self.basis = list(self.basis_map.values())
        # The number operators for each site in our chosen basis
        self.number_operators = self.get_number_operators()
        # The Hamiltonian matrix given the above parameters
        self.H = self.get_hamiltonian()
        # The density matrix of the ground state of this Hamiltonian
        self.rho_0 = self.get_ground_density()

    def get_basis_map(self):
        """Construct a dictionary mapping from bra-ket notation to the vector
           representation of all possible states.

        :returns: A mapping from states to their vector representation
        :rtype: dict
        """
        # Set the first possible basis state, which has the first ne sites
        # occupied
        state = np.zeros(self.L, dtype = int)
        for i in range(self.ne):
            state[i] = 1
        # Use permutations to get all possible states and store in a set to
        # avoid duplicates, which the permutations function will return
        state_set = set()
        perms = permutations(state)
        # Add each state to the set, automatically excluding duplicates
        for perm in perms:
            state_set.add(perm)
        # Cast to list for sorting. This gives us a known assignment of
        # basis states to basis vectors
        state_list = list(state_set)
        state_list.sort(reverse = True)
        # Associate each state to an explicit vector representation and create a
        # mapping between the state representation and vector representation
        n_states = len(state_list)
        basis_map = {}
        for i, state in enumerate(state_list):
            # create a vector representation for this state
            b = np.zeros(n_states)
            b[i] = 1
            basis_map[state] = b

        return basis_map

    def H_v(self, state):
        """Calculate the sum over sites associated with the interaction term of
           the get_hamiltonian.

        :param state: A tuple representing a state in our chosen basis
        :type state: tuple
        :returns: The associated sum
        :rtype: float
        """
        s = 0
        # Up to range -1 as we are taking the product with the nearest neighbour to
        # the right, and the last site has no neighbour to the right since we do not
        # have and boundary conditions yet
        for i in range(len(state)-1):
            s += state[i+1]*state[i]
        return s

    def H_t(self, state_i, state_j):
        """Calculate the kinetic term between state_i and state_j of the
           hamiltonian by determining whether state_i can be obtained by
           'hopping' from state_j.

        :param state_i: A tuple representing a state
        :type state: tuple
        :param state_i: A tuple representing a state
        :type state: tuple
        :returns: 1 if hopping is permitted, 0 otherwise
        :rtype: int
        """
        # The index of the last atom in the chain
        last_i = len(state_j)-1
        # Store state_j in another variable as we will change the variable state_j
        # in the for loop
        STATE_J = state_j
        # Cast state_i to a list for comparison with state_j, which will be cast to
        # a list below
        state_i = list(state_i)

        # i loops over every site in the chain
        for i in range(len(state_j)):
            state_j = list(STATE_J)
            # Check first site, can only hop to the right
            if i == 0:
                # This line checks if a hop to the right is permitted
                if ((state_j[0] == 1) and (state_j[1] == 0)):
                    # If it is, then it executes the hop and checks against state_i
                    state_j[0] = 0
                    state_j[1] = 1
                    if state_j == state_i:
                        return 1
            # Check last site, can only hop to teh left
            elif i == last_i:
                 # This line checks if a hop to the left is permitted
                if ((state_j[last_i] == 1) and (state_j[last_i-1] == 0)):
                    # If it is, then it executes the hop and checks against state_i
                    state_j[last_i] = 0
                    state_j[last_i-1] = 1
                    if state_j == state_i:
                        return 1
            # Check all others, can hop right or left
            else:
                # This line checks if a hop to the left is permitted
                if ((state_j[i] == 1) and (state_j[i-1] == 0)):
                    # If it is, then it executes the hop and checks against state_i
                    state_j[i] = 0
                    state_j[i-1] = 1
                    if state_j == state_i:
                        return 1
                    else:
                        # If not, reset state_j to check a hop to the right
                        state_j = list(STATE_J)
                # This line checks if a hop to the right is permitted
                if ((state_j[i] == 1) and (state_j[i+1] == 0)):
                    # If it is, then it executes the hop and checks against state_i
                    state_j[i] = 0
                    state_j[i+1] = 1
                    if state_j == state_i:
                        return 1

        return 0

    def get_hamiltonian(self):
        """Construct the Hamiltonain matrix for the Hubbard model.

        :returns: a numpy array representing the get_hamiltonian matrix
        :rtype: numpy.ndarray
        """

        # Define an empty matrix for the get_hamiltonian
        H = np.zeros(self.n_states*self.n_states).reshape(self.n_states, self.n_states)

        # Populate the get_hamiltonian
        for i in range(self.n_states):
            for j in range(self.n_states):
                # H_atom
                # I believe that we are casting to list here as each element in
                # the keys variable is a tuple?
                H[i, j] += (self.basis[i]@self.basis[j])*(self.E@np.array(list(self.states[i])))
                # H_mb
                H[i, j] += (self.basis[i]@self.basis[j])*self.V*self.H_v(self.states[i])
                # T
                if i != j:
                    H[i, j] += self.t*self.H_t(self.states[i], self.states[j])

        return H

    def solve(self):
        """Solve for the eigenvalues and eigenvectors of a given get_hamiltonian.

        :returns: Two numpy arrays, representing the eigenvalues and eigenvectors
        :rtype: numpy.ndarray
        """
        e_vals, e_vecs = linalg.eig(self.H)

        return e_vals, e_vecs

    def get_ground_state(self):
        """Find the ground state eignevector of the system."""
        # Solve for the eigenvectors and eigenvalues
        e_vals, e_vecs = self.solve()
        # Extract the ground state vector
        return e_vecs[:, np.argmin(e_vals)]

    def get_ground_density(self):
        """Find the density matrix of the ground state eignevector of the system."""
        # Solve for the eigenvectors and eigenvalues
        gs = self.get_ground_state()
        return np.outer(gs, gs)

    def get_number_operators(self):
        """Construct a collection of number operators in the selected basis.

        :returns: A collection of number operators
        :rtype: numpy.array(numpy.ndarray)
        """
        # Create an empty array to store the number operators
        number_operators = np.array([np.zeros(self.n_states*self.n_states).reshape(self.n_states,
                                    self.n_states) for i in range(self.L)])
        # i indexes each site/number operator
        for i in range(self.L):
            # Add a 1 to the diagonal of the number operator if the selected state
            # has site i occupied
            for j, state in enumerate(self.states):
                if state[i] == 1:
                    (number_operators[i])[j, j] = 1

        return number_operators
