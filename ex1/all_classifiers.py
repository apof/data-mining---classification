from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import pandas as pd

def KfoldCrossValidation(text_clf, folds, data):
	k_fold = KFold(n_splits=folds)
	accuracy = 0
	precision = 0
	recall = 0
	fmeasure = 0
	for train_indices, test_indices in k_fold.split(data['Category']):
		text_clf.fit(data['Content'][train_indices], data['Category'][train_indices])

		predicted = text_clf.predict(data['Content'][test_indices])

		accuracy += metrics.accuracy_score(data['Category'][test_indices], predicted)  
		#precision += metrics.precision_score(data['Category'][test_indices], predicted, average='macro')
		#recall += metrics.recall_score(data['Category'][test_indices], predicted, average='macro')
		#fmeasure += metrics.f1_score(data['Category'][test_indices], predicted, average='macro')

	print "Precision = " + str(precision/10)
	print "Accuracy = " + str(accuracy/10)
	print "Recall = " + str(recall/10)
	print "F-Measure = " + str(fmeasure/10)

def main():
	print "Main"
	data = pd.read_csv('./datasets/train_set.csv', sep="\t")
	folds = 10
	data = data[0:5000]

	svd_model = TruncatedSVD(n_components=500, algorithm='randomized',n_iter=10, random_state=42)

	#Random Forest
 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', RandomForestClassifier())])
 	#KfoldCrossValidation(text_clf, folds, data)

 	#SVM
 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', svm.SVC(C=1.0, kernel='linear',gamma='auto'))])
	#KfoldCrossValidation(text_clf, folds, data)

 	#Naive Bayes
 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', MultinomialNB())])
 	#KfoldCrossValidation(text_clf, folds, data)

 	#Stochastic Gradient Descent
 	#text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=2000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))])

 	#Stochastic Gradient Descent
 	#text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1))])
 	#KfoldCrossValidation(text_clf, folds, data)

if __name__ == "__main__":
	main()


# data = pd.read_csv('./datasets/train_set.csv', sep="\t")
# train_data = data
# content = data['Content']
# category = data['Category']
# title = data['Title']
# average_prec = 0
# accuracy = 0
# precision = 0
# recall = 0
# fmeasure = 0

# k_fold = KFold(n_splits=10)
# for train_indices, test_indices in k_fold.split(train_data['Category']):
# 	#print('Train: %s | test: %s' % (train_indices, test_indices))
	
# 	#Random Forest
# 	#text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('clf', RandomForestClassifier())])
	
# 	#SVM
# 	#text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))])
	
# 	#Naive Bayes
# 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('clf', MultinomialNB())])

# 	text_clf.fit(train_data['Content'][train_indices], train_data['Category'][train_indices])

# 	predicted = text_clf.predict(train_data['Content'][test_indices])

# 	accuracy += metrics.accuracy_score(train_data['Category'][test_indices], predicted)  
# 	precision += metrics.precision_score(train_data['Category'][test_indices], predicted, average='macro')
# 	recall += metrics.recall_score(train_data['Category'][test_indices], predicted, average='macro')
# 	fmeasure += metrics.f1_score(train_data['Category'][test_indices], predicted, average='macro')

# print "Precision = " + str(precision/10)
# print "Accuracy = " + str(accuracy/10)
# print "Recall = " + str(recall/10)
# print "F-Measure = " + str(fmeasure/10)