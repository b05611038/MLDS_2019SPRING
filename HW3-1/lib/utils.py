import pickle
import numpy as np
  
def save_object(fname, obj):
    #the function is used to save some data in class or object in .pkl file
    with open(fname, 'wb') as out_file:
        pickle.dump(obj, out_file)
    out_file.close()

def load_object(fname):
    #the function is used to read the data in .pkl file
    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)

def orthogonal(noise):
    shape = noise.shape
    noise = noise.reshape(shape[0], -1)
    mean = np.mean(noise, axis = 0)
    var = noise - mean
    length = np.linalg.norm(noise, axis = 0)

    alt_cov = np.dot(var, np.transpose(var))
    eig_val, eig_vec = np.linalg.eig(alt_cov)
    eig_vec = np.dot(np.transpose(var), eig_vec).astype(np.float64)
    eig_vec = np.multiply(length, (eig_vec / np.linalg.norm(eig_vec, axis = 0)))

    return eig_vec.reshape(shape)


