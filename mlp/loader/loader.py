import pickle
import os

def load(filename):
    with open(os.path.join('..', 'generated', filename), 'rb') as file:
        return pickle.load(file)