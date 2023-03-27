import spacy
from spacy import displacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import joblib

class SentimentAnalyser:

    nlp = spacy.load('en_core_web_sm')
    stopwords = list(STOP_WORDS)
    punct = string.punctuation

    def __init__(self):
        print("Do nothing")

    def text_data_cleaning(self, sentence):

        doc = self.nlp(sentence)

        tokens = []
        for token in doc:
            if token.lemma_ != "-PRON-":
                temp = token.lemma_.lower().strip()
            else:
                temp = token.lower_
            tokens.append(temp)

        cleaned_tokens = []

        for token in tokens:
            if token not in self.stopwords and token not in self.punct:
                cleaned_tokens.append(token)
        return cleaned_tokens

    def createModel(self):
        data_amazon = pd.read_csv(r'C:\Users\KIIT\Documents\NLP\amazon_cells_labelled.txt', sep = '\t', header = None)
        columns_name = ['Review', 'Sentiment']
        data_amazon.columns = columns_name

        print(data_amazon.head())
        print(data_amazon['Sentiment'].value_counts())
        print(data_amazon.isnull().sum())


        # print(self.text_data_cleaning("    Hello how are you"))
        tfidf = TfidfVectorizer(tokenizer = self.text_data_cleaning)
        classifier = LinearSVC()


        X = data_amazon['Review']
        y = data_amazon['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # print(X_train.shape, X_test.shape)
        clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        joblib.dump(clf, "TrainedModel.sav")

    def predictSentiment(self, sentiment):
        loaded_model = joblib.load("TrainedModel.sav")
        return loaded_model.predict([sentiment])


#createModel()
