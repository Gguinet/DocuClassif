import pickle
import json
from params import EURALEX_LABELS
from datasets import load_dataset

# load the model from disk
model = pickle.load(open('models/svm/svm_1.sav', 'rb'))
mlb = pickle.load(open('models/svm/mlb_1.sav', 'rb'))


# Load Eurovoc Concept
with open('data/eurovoc_descriptors.json') as jsonl_file:
    eurovoc_descriptors =  json.load(jsonl_file)

def predict_labels_svm(premise):

    predictions = model.predict([premise])
    pred_labels = [eurovoc_descriptors[EURALEX_LABELS[i]]['en'] 
                    for i in mlb.inverse_transform(predictions)[0]]

    return {"labels":pred_labels, "scores":None}
