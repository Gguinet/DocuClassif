# DocuClassif
Document Classification Pipeline.

Models:
- Zero-Shot Classification with Large Language Models (LLM)
- Support Vector Machine, using TD-IDF of n-grams as features.

Datasets for Training:
- [Multi-Eurlex](https://huggingface.co/datasets/multi_eurlex#dataset-structure)

Ideas of improvement:
- (ML) Few-shot Learning adding a last layer of prediction for the label class. 
- (Soft. Eng.) Dockerize the Python Server, following for instance [1](https://chatbotslife.com/deploying-transformer-models-1350876016f)
- Extend to [Lex Glue Dataset](https://huggingface.co/datasets/lex_glue)
