import logging
import argparse
import pickle
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--dataset',  default='multi_eurlex')
    parser.add_argument('--text_limit', default=2000)
    parser.add_argument('--n_classes', default=21)
    parser.add_argument('--training_size', default=-1)
    parser.add_argument('--save_best', default=True)


    config = parser.parse_args()

    handlers = [logging.FileHandler(f'logs/{config.dataset}/svm.txt'), logging.StreamHandler()]
    logging.basicConfig(handlers=handlers, level=logging.INFO)

    def get_text(data):
        return [' '.join(text.split()[:config.text_limit]) for text in data['text']]

    def get_labels(data, mlb):
        return mlb.transform(data['labels']).tolist()

    def add_zero_class(labels):
        augmented_labels = np.zeros((len(labels), len(labels[0]) + 1), dtype=np.int32)
        augmented_labels[:, :-1] = labels
        augmented_labels[:, -1] = (np.sum(labels, axis=1) == 0).astype('int32')
        return augmented_labels

    scores = {'micro-f1': [], 'macro-f1': []}
    dataset = load_dataset(config.dataset,'en', split='train')[:config.training_size]

    for seed in range(1, 6):
        classifier = OneVsRestClassifier(LinearSVC(random_state=seed, max_iter=50000))
        parameters = {
            'vect__max_features': [10000, 20000, 40000],
            'clf__estimator__C': [0.1, 1, 10],
            'clf__estimator__loss': ('hinge', 'squared_hinge')
        }
        
        # Init Pipeline (TF-IDF, SVM)
        text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('english'),
                                                      ngram_range=(1, 3), min_df=5)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', classifier),
                             ])

        # Hyper-param Optimization
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=32, verbose=4)

        # Pre-process inputs, outputs
        x_train = get_text(dataset) 
        mlb = MultiLabelBinarizer(classes=range(config.n_classes))
        mlb.fit(dataset['labels'])

        y_train = get_labels(dataset, mlb)

        # Train classifier
        gs_clf = gs_clf.fit(x_train, y_train)

        # Print best hyper-parameters
        logging.info('Best Parameters:')
        for param_name in sorted(parameters.keys()):
            logging.info("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

        dataset_val = load_dataset(config.dataset,'en', split='train')

        # Report Validation results
        dataset_val = load_dataset(config.dataset,'en', split='train')

        logging.info('VALIDATION RESULTS:')
        y_pred = gs_clf.predict(get_text(dataset_val))
        y_true = get_labels(dataset_val, mlb)
        y_true = add_zero_class(y_true)
        y_pred = add_zero_class(y_pred)

        logging.info(f'Micro-F1: {metrics.f1_score(y_true, y_pred, average="micro")*100:.1f}')
        logging.info(f'Macro-F1: {metrics.f1_score(y_true, y_pred, average="macro")*100:.1f}')

        # Report Test results
        dataset_test = load_dataset(config.dataset,'en', split='test')

        logging.info('TEST RESULTS:')
        y_pred = gs_clf.predict(get_text(dataset_test))
        y_true = get_labels(dataset_test, mlb)
        y_true = add_zero_class(y_true)
        y_pred = add_zero_class(y_pred)
        logging.info(f'Micro-F1: {metrics.f1_score(y_true, y_pred, average="micro")*100:.1f}')
        logging.info(f'Macro-F1: {metrics.f1_score(y_true, y_pred, average="macro")*100:.1f}')

        scores['micro-f1'].append(metrics.f1_score(y_true, y_pred, average="micro"))
        scores['macro-f1'].append(metrics.f1_score(y_true, y_pred, average="macro"))

        if config.save_best == True:

            pickle.dump(gs_clf, open(f'models/svm/svm_{seed}.sav', 'wb')) 
            pickle.dump(mlb, open(f'models/svm/mlb_{seed}.sav', 'wb')) 

    # Report averaged results across runs
    logging.info('-' * 100)
    logging.info(f'Micro-F1: {np.mean(scores["micro-f1"])*100:.1f} +/- {np.std(scores["micro-f1"])*100:.1f}\t'
                 f'Macro-F1: {np.mean(scores["macro-f1"])*100:.1f} +/- {np.std(scores["macro-f1"])*100:.1f}')


if __name__ == '__main__':
    main()