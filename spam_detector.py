import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


def load_dataset():
    candidates = ["spam.csv", "SMSSpamCollection", "sample_sms.csv"]
    for fname in candidates:
        if os.path.exists(fname):
            try:
                if fname == "SMSSpamCollection":
                    df = pd.read_csv(fname, sep='\t', header=None, names=["label", "message"], encoding='latin-1')
                else:
                    df = pd.read_csv(fname, encoding='latin-1')
            except Exception:
                df = pd.read_csv(fname, sep='\t', header=None, names=["label", "message"], encoding='latin-1')

            # Normalize common column names
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
            elif 'label' in df.columns and 'message' in df.columns:
                df = df[['label', 'message']]
            else:
                df = df.iloc[:, :2]
                df.columns = ['label', 'message']

            df = df.dropna()
            return df
    raise FileNotFoundError("No dataset file found. Add 'spam.csv', 'SMSSpamCollection', or 'sample_sms.csv' to the project folder.")


def preprocess(df):
    df = df.copy()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df = df[df['label'].isin(['spam', 'ham'])]
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    X_text = df['message'].astype(str)
    y = df['label_num'].values
    return X_text, y


def train_and_evaluate(X_text, y):
    vect = CountVectorizer(stop_words='english')
    X = vect.fit_transform(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return vect, clf


def predict_message(vect, clf, message: str):
    vec = vect.transform([message])
    pred = clf.predict(vec)[0]
    label = 'spam' if pred == 1 else 'ham'
    print(f"Message: {message}\nPrediction: {label} ({pred})")


def main():
    parser = argparse.ArgumentParser(description='Train and run a simple SMS spam detector')
    parser.add_argument('--predict', '-p', type=str, help='A message to classify (predict only after training)')
    args = parser.parse_args()

    print('Loading dataset...')
    df = load_dataset()
    X_text, y = preprocess(df)
    print(f'Dataset loaded: {len(X_text)} messages')

    print('Training model...')
    vect, clf = train_and_evaluate(X_text, y)

    if args.predict:
        predict_message(vect, clf, args.predict)


if __name__ == '__main__':
    main()
