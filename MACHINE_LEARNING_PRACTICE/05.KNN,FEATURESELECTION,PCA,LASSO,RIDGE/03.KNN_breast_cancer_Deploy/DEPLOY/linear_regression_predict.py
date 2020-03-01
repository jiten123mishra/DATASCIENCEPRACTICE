import pickle as pk
import numpy as np


def getPrediction(modelFileName, message1, message2, message3,message4, message5, message6, message7, message8, message9):
    
    load_model = pk.load(open(modelFileName, 'rb'))
    result=load_model.predict([[float(message1),float(message2),float(message3),float(message4),float(message5),float(message6),float(message7),float(message8),float(message9)]])
    #result=load_model.predict([[5.1,4.2,4.3,5.2,7.1,10.5,3.7,2.9,1.3]])
    return np.array_str(result)
    