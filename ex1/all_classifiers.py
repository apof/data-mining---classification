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
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
import math
from operator import itemgetter
import numpy as np
import pandas as pd

def KfoldCrossValidation(text_clf, folds, data):
	k_fold = KFold(n_splits=folds)
	accuracy = 0
	precision = 0
	recall = 0
	fmeasure = 0
	count = 1
	for train_indices, test_indices in k_fold.split(data['Category']):
		text_clf.fit(data['Content'][train_indices], data['Category'][train_indices])

		predicted = text_clf.predict(data['Content'][test_indices])

		accuracy += metrics.accuracy_score(data['Category'][test_indices], predicted)  
		#precision += metrics.precision_score(data['Category'][test_indices], predicted, average='macro')
		#recall += metrics.recall_score(data['Category'][test_indices], predicted, average='macro')
		#fmeasure += metrics.f1_score(data['Category'][test_indices], predicted, average='macro')

		print str(count) + "st fold completed"
		count += 1

	print "Precision = " + str(precision/10)
	print "Accuracy = " + str(accuracy/10)
	print "Recall = " + str(recall/10)
	print "F-Measure = " + str(fmeasure/10)

def KNN_CrossValidation(folds, data):
	k_fold = KFold(n_splits=folds)
	accuracy = 0
	precision = 0
	recall = 0
	fmeasure = 0
	count = 1
	for train_indices, test_indices in k_fold.split(data['Category']):
		#text_clf.fit(data['Content'][train_indices], data['Category'][train_indices])

		predicted = KNN(data['Content'][train_indices], data['Category'][train_indices], data['Content'][test_indices])
		predicted = np.asarray(predicted)
		accuracy += metrics.accuracy_score(data['Category'][test_indices], predicted) 
		precision += metrics.precision_score(data['Category'][test_indices], predicted, average='macro')
		recall += metrics.recall_score(data['Category'][test_indices], predicted, average='macro')
		fmeasure += metrics.f1_score(data['Category'][test_indices], predicted, average='macro')

		print str(count) + "st fold completed"
		count += 1

	print "Precision = " + str(precision/10)
	print "Accuracy = " + str(accuracy/10)
	print "Recall = " + str(recall/10)
	print "F-Measure = " + str(fmeasure/10)

def KNN(train_data, target, predict_data):
	
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)
	X = vectorizer.fit_transform(train_data)
	n_comp = 200
	svd = TruncatedSVD(algorithm='randomized', n_components=n_comp, n_iter=7, random_state=42, tol=0.0)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)
	X = lsa.fit_transform(X)
	
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)
	X_predict = vectorizer.fit_transform(predict_data)
	svd = TruncatedSVD(algorithm='randomized', n_components=n_comp, n_iter=7, random_state=42, tol=0.0)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)
	X_predict = lsa.fit_transform(X_predict)

	#print X_predict

	out = []
	#for i in range(X_predict.shape[0]):
		#distances = calculate_distances(X_predict[i], X, target)
	for i in range(len(X_predict)):
		distances = calculate_distances(X_predict[i], X, target)
		hashmap = {}
		hashmap['Politics'] = 0
		hashmap['Film'] = 0
		hashmap['Football'] = 0
		hashmap['Business'] = 0
		hashmap['Technology'] = 0
		for i in range(7):
			hashmap[distances[i][1]] += 1
		sorted_hashmap = sorted(hashmap.items(), key=itemgetter(1),reverse=True)
		out.append(sorted_hashmap[0][0])

	return out


def calculate_distances(X_predict, X, target):

	target = target.values;
	
	####################
	#Eycleidian Distance
	####################
	# distances = []
	# for i in range(len(X)):
	# 	distance = 0
	# 	for j in range(len(X[i])):
	# 		if(j >= len(X_predict)):
	# 			distance += X[i][j]*X[i][j]
	# 		else:
	# 			distance += (X_predict[j]-X[i][j])*(X_predict[j]-X[i][j])
	# 	if(len(X_predict) > len(X[i])):
	# 		for j in range(len(X[i]),len(X_predict)):
	# 			distance += X_predict[j]*X_predict[j]
	# 	distances.append((math.sqrt(distance), target[i]))
	# sorted_distances = sorted(distances,key=itemgetter(0))

	##################
	#Cosine Similarity
	##################
	norm_b = LA.norm(X_predict)
	distances = []
	for i in range(len(X)):
		norm_a = LA.norm(X[i])
		distance = 0
		for j in range(len(X[i])):
			if(j >= len(X_predict)):
				distance += 0
			else:
				distance += X_predict[j]*X[i][j]
		distances.append((distance/(norm_a*norm_b), target[i]))
	sorted_distances = sorted(distances,key=itemgetter(0),reverse=True)

	
	return sorted_distances
		
def append_title_to_content(data, times_to_append_title):
	for i in range(len(data['Title'])):
		strr=""
		for j in range(times_to_append_title):
			#data['Content'][i] += data['Title'][i];
			strr += data['Title'][i];
		data['Content'][i] += strr
	print "Title preprocessing is done"

def main():
	data = pd.read_csv('./datasets/train_set.csv', sep="\t")
	folds = 10
	data = data[0:1000]
	append_title_to_content(data,20)

	svd_model = TruncatedSVD(n_components=500, algorithm='randomized',n_iter=10, random_state=42)

	#add_stop_words = ["say","said","will","one"]
	#enhanced_stop_words = ENGLISH_STOP_WORDS.union(add_stop_words)


	#KNN
	#KNN_CrossValidation(folds, data)

	#Random Forest
 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', RandomForestClassifier())])
 	#KfoldCrossValidation(text_clf, folds, data)

 	#SVM-Grid Search
 	#params = {'clf__C': [1, 10, 100, 1000], 'clf__gamma': [0.001, 0.0001], 'clf__kernel': ['linear', 'rbf']}
 	#text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', svm.SVC())])
 	#grid = GridSearchCV(estimator=text_clf, param_grid=params, n_jobs=-1)
 	#grid.fit(data['Content'], data['Category'])
 	#print("Best score: %0.3f" % grid.best_score_)
   	#print("Best parameters set:")
   	#best_parameters = grid.best_estimator_.get_params()
   	#for param_name in sorted(params.keys()):
	#	print("\t%s: %r" % (param_name, best_parameters[param_name]))
 	
 	#SVM with the best hyperparameters
 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', svm.SVC(C=1.0, kernel='linear',gamma=0.001))])
	#KfoldCrossValidation(text_clf, folds, data)

 	#Naive Bayes
 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', MultinomialNB())])
 	#KfoldCrossValidation(text_clf, folds, data)

 	#Stochastic Gradient Descent
 	text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=2000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))])
 	KfoldCrossValidation(text_clf, folds, data)

 	#Stochastic Gradient Descent
 	#text_clf = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)), ('tfidf', TfidfTransformer()),('svd', svd_model),('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1))])
 	#KfoldCrossValidation(text_clf, folds, data)

if __name__ == "__main__":
	main()