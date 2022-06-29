# DocuClassif: A Document Classification Pipeline

### Models:
- Zero-Shot Classification with Large Language Models (LLM)
- Support Vector Machine, using TD-IDF of n-grams as features.

### Dataset used for Training/Testing:
- [Multi-Eurlex](https://huggingface.co/datasets/multi_eurlex#dataset-structure)

### Use of API:

Locally, the API can be called by using the following instructions:
```
curl -X 'POST' 'http://127.0.0.1:8000/classification' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"text":"Make decisions 10X more confidently and quickly with AI-powered insights.", \
         "labels":["Business","Machine Learning","Sport"], \
         "model": "zero-shot"}'
```
The ouput is a dict with the predicted labels. ```model``` is to be chosen in ```['svm','zero-shot']```. ```labels``` is an option input if the ```zero-shot``` model is used (no specification leads to a default list specified in  ```params.py```). Note that SVM model only works with the EuraLex Labels (specified in  ```params```).

### File Structure:
- ```main.py``` is the API structure.
-  ```params.py``` contains the main Hyperparameter.
-  ```test.py``` is an inner testing file to to compare performance of models.
- For the SVM Model, 
  - ```svm_classif.py``` is the SVM classifier trained on Multi-Eurlex and stored in ```models/svm```. 
  - Note that ```svm_classif_training.py``` allows to train the SVM model on a given dataset (e.g. EuraLex) while performing a Grid Search for Hyperparameter Tunning.
-  ```zero_classif.py``` is the Zero-Shot NLI Classifier imported from Hugging Face Library (default is ```facebook/bart-large-mnli```).


### Ideas of improvement:
- [ ] (ML) Few-shot Learning adding a last layer of prediction for the label class. 
- [ ] (Soft. Eng.) Dockerize the Python Server, following for instance [1](https://chatbotslife.com/deploying-transformer-models-1350876016f)
- [ ] Extend to [Lex Glue Dataset](https://huggingface.co/datasets/lex_glue)