# DocuClassif
Document Classification Pipeline

Datasets:
- [https://huggingface.co/datasets/multi_eurlex#dataset-structure]

Models:
- Zero-Shot Classification with Large Language Models (LLM)
- Support Vector Machine, using TD-IDF of n-grams as features.

Ideas of improvement:
- (ML) Few-shot Learning adding a last layer of prediction for the label class. 
- (Software) Dockerize the Python Server, following for instance [https://chatbotslife.com/deploying-transformer-models-1350876016f]
- Extend to [https://huggingface.co/datasets/lex_glue]