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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
import math
from operator import itemgetter
import numpy as np
import pandas as pd

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

def unseen_data_predict_combine_3classifiers(text_clf1, text_clf2, text_clf3, train_data, unseen_data):
	category_map = []
	category_map.append("Business")
	category_map.append("Film")
	category_map.append("Football")
	category_map.append("Politics")
	category_map.append("Technology")

	text_clf1.fit(train_data['Content'], train_data['Category'])
	predicted1 = text_clf1.predict(unseen_data['Content'])
	y = text_clf1.predict_proba(unseen_data['Content'])
	text_clf1.fit(train_data['Title'], train_data['Category'])
	y1 = text_clf1.predict_proba(unseen_data['Title'])
	for i in range(len(y)):
		mm = max(y[i]+y1[i])
		index = np.where((y[i]+y1[i]) == mm)
		predicted1[i] = category_map[index[0][0]]

	text_clf2.fit(train_data['Content'], train_data['Category'])
	predicted2 = text_clf2.predict(unseen_data['Content'])
	y = text_clf2.predict_proba(unseen_data['Content'])
	text_clf2.fit(train_data['Title'], train_data['Category'])
	y1 = text_clf2.predict_proba(unseen_data['Title'])
	for i in range(len(y)):
		mm = max(y[i]+y1[i])
		index = np.where((y[i]+y1[i]) == mm)
		predicted2[i] = category_map[index[0][0]]

	text_clf3.fit(train_data['Content'], train_data['Category'])
	predicted3 = text_clf3.predict(unseen_data['Content'])
	y = text_clf3.predict_proba(unseen_data['Content'])
	text_clf3.fit(train_data['Title'], train_data['Category'])
	y1 = text_clf3.predict_proba(unseen_data['Title'])
	for i in range(len(y)):
		mm = max(y[i]+y1[i])
		index = np.where((y[i]+y1[i]) == mm)
		predicted3[i] = category_map[index[0][0]]
	
	for j in range(len(predicted1)):
		hashmap = {}
		hashmap['Politics'] = 0
		hashmap['Film'] = 0
		hashmap['Football'] = 0
		hashmap['Business'] = 0
		hashmap['Technology'] = 0
		#print hashmap[predicted1[j]]
		hashmap[predicted1[j]] += 1
		hashmap[predicted2[j]] += 1
		hashmap[predicted3[j]] += 1
		sorted_hashmap = sorted(hashmap.items(), key=itemgetter(1),reverse=True)
		if(sorted_hashmap[0][1] > 1):
			#if(predicted1[j] != sorted_hashmap[0][0]):
				#print "changed!!!"
			predicted1[j] = sorted_hashmap[0][0]

	file = open("./testSet_categories.csv",'w')
	file.write("Id"+","+"Category"+"\n")
	for i in range(len(predicted1)):
		file.write(str(unseen_data['Id'][i]))
		file.write(",")
		file.write(predicted1[i])
		file.write("\n")
	file.close()
	

def KfoldCrossValidation(text_clf, folds, data):
	k_fold = KFold(n_splits=folds)
	accuracy = 0
	precision = 0
	recall = 0
	fmeasure = 0
	count = 1

	category_map = []
	category_map.append("Business")
	category_map.append("Film")
	category_map.append("Football")
	category_map.append("Politics")
	category_map.append("Technology")

	for train_indices, test_indices in k_fold.split(data['Category']):
		text_clf.fit(data['Content'][train_indices], data['Category'][train_indices])

		predicted = text_clf.predict(data['Content'][test_indices])

		y= text_clf.predict_proba(data['Content'][test_indices])
		
		text_clf.fit(data['Title'][train_indices], data['Category'][train_indices])
		y1= text_clf.predict_proba(data['Title'][test_indices])
		
		for i in range(len(y)):
			mm = max(y[i]+y1[i])
			index = np.where((y[i]+y1[i]) == mm)
			predicted[i] = category_map[index[0][0]]
			
		#print predicted
		#use_title_for_prediction(predicted,data,test_indices)
		#print predicted
		accuracy += metrics.accuracy_score(data['Category'][test_indices], predicted)  
		#precision += metrics.precision_score(data['Category'][test_indices], predicted, average='macro')
		#recall += metrics.recall_score(data['Category'][test_indices], predicted, average='macro')
		#fmeasure += metrics.f1_score(data['Category'][test_indices], predicted, average='macro')

		print str(count) + "st fold completed"
		print metrics.accuracy_score(data['Category'][test_indices], predicted)
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
		
def append_title_to_content(data, unseen_data, times_to_append_title):
	for i in range(len(data['Title'])):
		strr=""
		for j in range(times_to_append_title):
			#data['Content'][i] += data['Title'][i];
			strr += data['Title'][i];
		data['Content'][i] += strr

	for i in range(len(unseen_data['Title'])):
		strr=""
		for j in range(times_to_append_title):
			#unseen_data['Content'][i] += data['Title'][i];
			strr += unseen_data['Title'][i];
		unseen_data['Content'][i] += strr
	print "Title preprocessing is done"

def main():
	data = pd.read_csv('./datasets/train_set.csv', sep="\t")
	unseen_data = pd.read_csv('./datasets/test_set.csv', sep="\t")
	folds = 10
	#data = data[0:4090]
	#append_title_to_content(data,unseen_data,1)

	svd_model = TruncatedSVD(n_components=500, algorithm='randomized',n_iter=10, random_state=42)

	#add_stop_words = ["say","said","will","one","now","government","people","time","make","made","UK","US"]
	#enhanced_stop_words = ENGLISH_STOP_WORDS.union(add_stop_words)


	#KNN
	#KNN_CrossValidation(folds, data)

	#Random Forest---accuracy=93.54%
 	text_clf5 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', RandomForestClassifier())])
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
 	
 	#SVM with the best hyperparameters---accuracy=96.43% (n_components=1000,max_features=2000), accuracy=96.36% (n_components=300,max_features=2000),
 	#accuracy=96.48% (n_components=300,max_features=3000),  accuracy=96.56% (n_components=500,max_features=3000),
 	#accuracy=96.60% (n_components=300,max_features=5000)
 	#accuracy=96.61% (n_components=400,max_features=5000)
 	#accuracy=96.65% (n_components=500,max_features=5000),accuracy=96.72% (n_components=500,max_features=7000),accuracy=96.72% (n_components=500,max_features=8000),accuracy=96.68% (n_components=500,max_features=10000)
 	#accuracy=96.56% (n_components=600,max_features=5000)
 	#accuracy=96.60% (n_components=650,max_features=5000)
 	#accuracy=96.62% (n_components=750,max_features=5000)
 	#accuracy=96.62% (n_components=1000,max_features=5000)
 	text_clf1 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', svm.SVC(C=1.0, kernel='linear',gamma=0.001,probability=True))])
	#KfoldCrossValidation(text_clf1, folds, data)
	
	#accuracy=95.17% (n_components=500,max_features=7000)
 	#Naive Bayes
 	text_clf4 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
 	#KfoldCrossValidation(text_clf4, folds, data)

 	#Stochastic Gradient Descent---accuracy=96.38%(random hyperparameters)
 	text_clf2 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None, n_jobs=-1))])
 	#KfoldCrossValidation(text_clf2, folds, data)

 	#95.89%
 	#Neural Network Descent--accuracy=94.25%(10,20), accuracy=94.83%(20,20), accuracy=94.48%(20,30), accuracy=94.54%(10,30), accuracy=94.78%(30,30)
 	#accuracy with title: accuracy=96.2%(8,10), accuracy=96.3%(8,15), accuracy=96.28%(9,10)
 	text_clf3 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 10), random_state=1))])
 	KfoldCrossValidation(text_clf3, folds, data)


 	#unseen_data_predict(text_clf, folds, data, unseen_data)
 	#unseen_data_predict_combine_3classifiers(text_clf1,text_clf3,text_clf4,data,unseen_data)

if __name__ == "__main__":
	main()