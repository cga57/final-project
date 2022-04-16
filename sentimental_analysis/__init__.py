# Called to get a sentimental analysis model
# Tries to train a new one if one is not available

import pickle
from .sa_sklearn import *

model_path_pn = "sa_model_sklearn_svm_pn.pkl"
model_path_pnn = "sa_model_sklearn_svm_pnn.pkl"

def get_samodel_pn():
    # Check if a trained model exists
    try:
        print("Trying to load pretrained pn model from", model_path_pn)
        with open(model_path_pn, 'rb') as f:
            sa_model = pickle.load(f)

            return sa_model
    # If not, train the model and pass it
    except:
        # train model, save model, return model
        print("No Sentimental Analysis Model(PN) found. Training new one")
        sa_model = train_model_pn()

        print("Finished training model. Serializing to pickle and saving it for later")
        with open(model_path_pn, 'wb') as f:
            pickle.dump(sa_model, f)

        return sa_model

def get_samodel_pnn():
    #Check if a trained model exists
    try:
        print("Trying to load pretrained pnn model from", model_path_pnn)
        with open(model_path_pnn, 'rb') as f:
            sa_model = pickle.load(f)

            return sa_model
    # If not, train the model and pass it
    except:
        # train model, save model, return model
        print("No Sentimental Analysis Model(PNN) found. Training new one")
        sa_model = train_model_pnn()

        print("Finished training model. Serializing to pickle and saving it for later")
        with open(model_path_pnn, 'wb') as f:
            pickle.dump(sa_model, f)

        return sa_model
