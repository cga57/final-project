import pickle
from .sa_sklearn import *

model_path = "sa_model_sklearn_svm.pkl"

def get_samodel():
    # Check if a trained model exists
    try:
        print("Trying to load pretrained model from", model_path)
        with open(model_path, 'rb') as f:
            sa_model = pickle.load(f)

            return sa_model
    # If not, train the model and pass it
    except:
        # train model, save model, return model
        print("No Sentimental Analysis Model found. Training new one")
        sa_model = train_model()

        print("Finished training model. Serializing to pickle and saving it for \
            later")
        with open(model_path, 'wb') as f:
            pickle.dump(sa_model, f)

        return sa_model
