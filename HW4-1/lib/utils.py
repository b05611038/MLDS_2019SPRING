import os
import pickle

def save_object(fname, obj):
    #the function is used to save some data in class or object in .pkl file
    with open(fname, 'wb') as out_file:
        pickle.dump(obj, out_file)
    out_file.close()
    return None

def load_object(fname):
    #the function is used to read the data in .pkl file
    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)

def save_config(config, model_name, state):
    save_path = os.path.join('./output', model_name, model_name + '_s' + str(state) + '.config')
    save_object(save_path, config)
    return None


