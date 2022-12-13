import pandas as pd
import pickle

from sklearn import metrics


def predict(models):

    # Load prediction datasets
    fake = pd.read_csv('../data/fake_test.csv')


    # Setup predict data
    X_fake = fake.Text
    y_fake = fake.Score



    # Load TFIDF and transform
    vect = pickle.load(open('../train/models/tfidf.pkl', 'rb'))
    X_fake_dtm = vect.transform(X_fake)

    # Load classifier
    for mod in models:
        if mod=='tree':
            model = pickle.load(open(f'../train/models/{mod}.pkl', 'rb'))
        if mod=='dummy':
            model = pickle.load(open(f'../train/models/{mod}.pkl', 'rb'))
        if mod=='rf':
            model = pickle.load(open(f'../train/models/{mod}.pkl', 'rb'))

        print(f'*** EVAL {mod} ***')
        # Predict fake
        y_pred = model.predict(X_fake_dtm)
        acc=metrics.accuracy_score(y_fake, y_pred)
        print('acc fake= ',acc)
        # print(metrics.classification_report(y_fake, y_pred, digits=3))
        # Predict true
        if acc<0.4435:
            print('Nice this it better than baseline by ',0.4435-acc,' percentual points :D' )
        else:
            print('Performance worse than baseline by ',0.4435-acc,' percentual points')
if __name__ == "__main__":

    predict()