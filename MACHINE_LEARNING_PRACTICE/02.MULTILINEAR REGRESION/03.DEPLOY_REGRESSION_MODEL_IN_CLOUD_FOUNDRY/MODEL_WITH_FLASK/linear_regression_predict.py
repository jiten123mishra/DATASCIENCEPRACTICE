import pickle as pk
import numpy as np


def getPrediction(modelFileName, message1, message2, message3):
    
    load_model = pk.load(open(modelFileName, 'rb'))
    result=load_model.predict([[float(message1),float(message2),float(message3)]])
    return np.array_str(result)
    