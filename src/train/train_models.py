import pandas as pd
import pickle

from regex import F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier



def train_models(path,models):
    df=pd.read_csv(path)
    print('number of rows =',df.shape)

    # Train setup
    X = df.Text
    y = df.Score

    # Create an instance of TfidfVectorizer
    vect = TfidfVectorizer()

    # Fit to the data and transform to feature matrix
    X_train = vect.fit_transform(X)

    # Save model TFIDF
    pickle.dump(vect, open(b'../train/models/tfidf.pkl', "wb"))

    # Train classifier
    for mod in models:
        if mod=='tree':
            model = DecisionTreeClassifier(random_state=42)
        if mod=='dummy':
            model = DummyClassifier(strategy="most_frequent")
        if mod=='rf':
            model = RandomForestClassifier(n_jobs=-1)

        
        print(f'***  Training {mod} ***')
        model.fit(X_train, y)

        # Save classifier
        pickle.dump(model, open('../train/models/{}.pkl'.format(mod), "wb"))
    

if __name__ == "__main__":
    import sys
    sys.path.insert(1, '../predict')
    from predict import predict

    path='../data/true_train.csv'
    print(path)
    train_models(path)
    models=['catboost']
    predict(models)