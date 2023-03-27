from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
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
from analyzer import SentimentAnalyser


def home(request):
    return render(request, "analyzer/home.html", {'password':'uijib'})

def dataset(request):
    sent = str(request.GET.get('sentiment'))
    a = SentimentAnalyser.SentimentAnalyser()
    v = a.predictSentiment(sent)
    if v == 1:
        return render(request, "analyzer/Positive.html", {'result':v})
    elif v == 0:
        return render(request, "analyzer/Negative.html", {'result':v})
