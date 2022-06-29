import argparse
import pickle
import numpy as np
import json

from datasets import load_dataset
from sklearn import metrics
from svm_classif import predict_labels_svm
from zero_classif import predict_labels_zero
from params import EURALEX_LABELS,EURALEX_LABELS_LIST
from sklearn.preprocessing import MultiLabelBinarizer

def main():

    # Required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='multi_eurlex')
    parser.add_argument('--nb_pts', type=int, default=10)
    config = parser.parse_args()

    # Load Test Dataset
    dataset = load_dataset(config.dataset,'en', split='test')

    # Load Eurovoc Concept
    with open('data/eurovoc_descriptors.json') as jsonl_file:
        eurovoc_descriptors =  json.load(jsonl_file)

    def add_zero_class(labels):
        augmented_labels = np.zeros((len(labels), len(labels[0]) + 1), dtype=np.int32)
        augmented_labels[:, :-1] = labels
        augmented_labels[:, -1] = (np.sum(labels, axis=1) == 0).astype('int32')
        return augmented_labels
    
    def get_labels(pred, mlb):
        return mlb.transform([pred])[0]

    # Global Evaluation Variables
    mlb = MultiLabelBinarizer(classes=EURALEX_LABELS_LIST)
    mlb = mlb.fit(y=[EURALEX_LABELS_LIST])
    y_pred_svm,y_pred_zero,y_true = [],[],[]

    for i in range(min(len(dataset),config.nb_pts)):

        pred_labels = [eurovoc_descriptors[EURALEX_LABELS[i]]['en'] 
                        for i in dataset['labels'][i]]
        print('TRUE LABELS: {}'.format(pred_labels))

        pred_svm = predict_labels_svm(dataset['text'][i])['labels']
        print(f'PRED - SVM: {pred_svm}')

        pred_zero = predict_labels_zero(dataset['text'][i],
                                        labels=EURALEX_LABELS_LIST)['labels']
        print(f'PRED - Zero-Shot: {pred_zero}\n')

        y_pred_svm.append(get_labels(pred_svm, mlb))
        y_pred_zero.append(get_labels(pred_zero, mlb))
        y_true.append(get_labels(pred_zero, mlb))

    # Global Evaluation    
    y_true = add_zero_class(np.array(y_pred_svm))
    y_pred_svm = add_zero_class(np.array(y_pred_svm))
    y_pred_zero = add_zero_class(np.array(y_pred_zero))
    print(f'Micro-F1 - SVM: {metrics.f1_score(y_true, y_pred_svm, average="micro")*100:.1f}')
    print(f'Macro-F1 - SVM: {metrics.f1_score(y_true, y_pred_svm, average="macro")*100:.1f}')
    print(f'Micro-F1 - Zero-Shot: {metrics.f1_score(y_true, y_pred_zero, average="micro")*100:.1f}')
    print(f'Macro-F1 - Zero-Shot: {metrics.f1_score(y_true, y_pred_zero, average="macro")*100:.1f}')


if __name__ == '__main__':
    main()