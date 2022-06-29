import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from params import LABEL_LIST,MODEL_NAME,TOKENIZER_NAME

nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5

def single_classif(premise,label):

    hypothesis = f'This example is about {label}.'
    
    # run through model pre-trained on MNLI
    x = tokenizer.encode(premise, 
                         hypothesis,
                         return_tensors='pt',
                         truncation=True)
    logits = nli_model(x.to(device))[0]

    # We throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,1]

    return prob_label_is_true

def predict_labels_zero(premise,labels=LABEL_LIST):

    return {"labels":[label for label in labels if single_classif(premise,label)>= THRESHOLD]}





