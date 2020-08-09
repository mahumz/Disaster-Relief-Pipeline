import pickle
import re
import sys
import warnings
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine


def load_data(database_filepath, table_name='myDataTable'):
    """
    Args:
        database_filepath: Contains cleaned data table.
        table_name: Contains cleaned data.
    Returns:
       X: Disaster messages.
       Y: Disaster categories for each message.
       category_name: Disaster category names.
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, con=engine)

    category_names = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[category_names].values

    return X, y, category_names


def tokenize(text, lemmatizer=WordNetLemmatizer()):
    """Tokenize text (a disaster message).
    Args:
        text: String. A disaster message.
        lemmatizer: nltk.stem.Lemmatizer.
    Returns:
        list. It contains tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    tokens = [t for t in tokens if t not in stopwords.words('english')]

    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def save_stats(X, Y, category_names, vocab_filepath, category_filepath):
    """
    Args;
        X: Disaster messages.
        Y: Disaster categories for each messages.
        category_names: Disaster category names.
        vocab_filepath: Vocab is saved as pickle into this file.
        category_filepath: Category is saved as pickle into this file.
    """
    # Check vocabulary
    vect = CountVectorizer(tokenizer=tokenize)
    X_vectorized = vect.fit_transform(X)

    # Convert vocabs into pandas.dataframe
    keys, values = [], []
    for k, v in vect.vocabulary_.items():
        keys.append(k)
        values.append(v)
    vocab_df = pd.DataFrame.from_dict({'words': keys, 'counts': values})

    # Vocabulary stats
    vocab_df = vocab_df.sample(30, random_state=72).sort_values('counts', ascending=False)
    vocab_counts = list(vocab_df['counts'])
    vocab_words = list(vocab_df['words'])

    # Save vocaburaly stats
    with open(vocab_filepath, 'wb') as vocab_file:
        pickle.dump((vocab_counts, vocab_words), vocab_file)

    # Category stats
    category_counts = list(Y.sum(axis=0))

    # Save category stats
    with open(category_filepath, 'wb') as category_file:
        pickle.dump((category_counts, list(category_names)), category_file)


def build_model():
    """
    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a estimator.
    """
    # Set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                learning_rate=0.3,
                n_estimators=200
            )
        ))
    ])

    # Set parameters for gird search
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__n_estimators': [100, 200]
    }

    # Set grid search
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Args:
        model: sklearn.model_selection.GridSearchCV.  It contains a estimator.
        X_test: Disaster messages.
        Y_test: Disaster categories for each messages
        category_names: Disaster category names.
    """
    # Predict categories of messages.
    Y_pred = model.predict(X_test)

    # Print accuracy, precision, recall and f1_score for each categories
    for i in range(0, len(category_names)):
        print(category_names[i])
        print("\tAccuracy: {:.4f}\t\t% Precision: {:.4f}\t\t% Recall: {:.4f}\t\t% F1_score: {:.4f}".format(
            accuracy_score(Y_test[:, i], Y_pred[:, i]),
            precision_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            recall_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')
        ))


def save_model(model, model_filepath):
    """
    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a estimator.
        model_filepath: Trained model is saved as pickle into this file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 5:
        database_filepath, model_filepath, vocab_filepath, category_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Saving stats...')
        save_stats(X, Y, category_names, vocab_filepath, category_filepath)

        print('Building model...')
        model = build_model()

        print('Training model...')
        # Ignoring UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no true samples.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db test_classifier.pkl vocab.pkl category_pkl')


if __name__ == '__main__':
    main()