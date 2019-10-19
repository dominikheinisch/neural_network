import pickle
import os

def save(data, filename):
    with open(os.path.join('..', 'generated', filename), 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
