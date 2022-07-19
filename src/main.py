""" A main file to run the program from."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Helper import Hubbard, DataGenerator, Model, Tester





def main():
    """The main method."""
    L = 4
    ne = 2
    V = 2
    E = [0, 0, 0, 0]
    t_tot = 20
    dt = 0.001
    n_data = 10
    n_save = 1000
    memory = 1
    p_step = 1

    system = Hubbard(L=L, ne=ne, V=V, E=E)

if __name__ == '__main__':
    main()



