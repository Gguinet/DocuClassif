from fastapi import FastAPI
from pydantic import BaseModel, constr, validator
from typing import List
from params import LABEL_LIST
from svm_classif import predict_labels_svm
from zero_classif import predict_labels_zero


app = FastAPI(
    title="Document Classification API",
    description="A simple API that use NLP model to predict the class of a document.",
    version="1.0",
)

class UserRequestIn(BaseModel):
    text: constr(min_length=1)
    labels: List[str] = LABEL_LIST
    model: str
    @validator('model')
    def model_must_be_in_models(cls,model):
      models=['svm','zero-shot']
      if model not in models:
        raise ValueError(f'Model must be in {models}')
      return model

class LabelsOut(BaseModel):
    labels: List[str]
    #scores: List[float] = None

@app.post("/classification", response_model=LabelsOut)
def read_classification(user_request_in: UserRequestIn):

    if user_request_in.model == 'svm':
        return predict_labels_svm(user_request_in.text)

    elif user_request_in.model == 'zero-shot':
        return predict_labels_zero(user_request_in.text,
                                   user_request_in.labels)
    else:
        raise ValueError(f'No Model Specified')