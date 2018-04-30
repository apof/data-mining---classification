from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from operator import itemgetter
import math
import numpy as np
import pandas as pd
import KNN_classifier as knn
import cross_validation as cross
import beat_the_bench as beat
import string
from sklearn.base import TransformerMixin 
from spacy.en import English
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

def unseen_data_predict(text_clf, train_data, unseen_data):
	text_clf.fit(train_data['Content'], train_data['Category'])

	predicted = text_clf.predict(unseen_data['Content'])
	
	file = open("./testSet_categories.csv",'w')
	file.write("Id"+","+"Category"+"\n")
	for i in range(len(predicted)):
		file.write(str(unseen_data['Id'][i]))
		file.write(",")
		file.write(predicted[i])
		file.write("\n")
	file.close()

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Basic utility function to clean the text 
def clean_text(text):     
    return text.strip().lower()

def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in ENGLISH_STOP_WORDS and tok not in punctuations)]
    return tokens

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_tfidf_vect = StemmedCountVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)

def main():
	data = pd.read_csv('./datasets/train_set.csv', sep="\t")
	unseen_data = pd.read_csv('./datasets/test_set.csv', sep="\t")
	folds = 10

	svd_model = TruncatedSVD(n_components=500, algorithm='randomized',n_iter=10, random_state=42)

	#add_stop_words = ["say","said","will","one","review","new","will","US","UK"]
	#enhanced_stop_words = ENGLISH_STOP_WORDS.union(add_stop_words)

	#KNN
	#cross.KNN_CrossValidation(folds, data)

	#Random Forest---accuracy=93.54%
 	text_clf = Pipeline([('vect', stemmed_tfidf_vect),('svd', svd_model),('clf', RandomForestClassifier())])
 	#cross.KfoldCrossValidation(text_clf, folds, data)

 	#SVM with the best hyperparameters
 	#text_clf1 = Pipeline([('cleaner', predictors()),('vect', TfidfVectorizer(tokenizer = spacy_tokenizer,stop_words=ENGLISH_STOP_WORDS,max_features=2000)), ('svd', svd_model),('clf', svm.SVC(C=1.0, kernel='linear',gamma=0.001))])
 	#text_clf1 = Pipeline([('vect', TfidfVectorizer(tokenizer = stemming_tokenizer,stop_words=ENGLISH_STOP_WORDS,max_features=2000)), ('svd', svd_model),('clf', svm.SVC(C=1.0, kernel='linear',gamma=0.001))])
 	text_clf1 = Pipeline([('vect', stemmed_tfidf_vect),('svd', svd_model),('clf', svm.SVC(C=1.0, kernel='linear',gamma=0.001))])
	#cross.KfoldCrossValidation(text_clf1, folds, data)
	
 	#Naive Bayes
 	text_clf = Pipeline([('vect', stemmed_tfidf_vect),('clf', MultinomialNB())])
 	cross.KfoldCrossValidation(text_clf, folds, data)

 	#Stochastic Gradient Descent---accuracy=96.38%(random hyperparameters)
 	text_clf2 = Pipeline([('vect', stemmed_tfidf_vect),('svd', svd_model),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None, n_jobs=-1))])
 	#cross.KfoldCrossValidation(text_clf2, folds, data)


 	#Stochastic Gradient Descent--accuracy=94.25%(10,20), accuracy=94.83%(20,20), accuracy=94.48%(20,30), accuracy=94.54%(10,30), accuracy=94.78%(30,30)
 	text_clf3 = Pipeline([('vect', stemmed_tfidf_vect),('svd', svd_model),('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20), random_state=1))])
 	#cross.KfoldCrossValidation(text_clf3, folds, data)

 	#unseen_data_predict(text_clf, folds, data, unseen_data)
 	#beat.unseen_data_predict_combine_3classifiers(text_clf1,text_clf2,text_clf3,data,unseen_data)

if __name__ == "__main__":
	main()