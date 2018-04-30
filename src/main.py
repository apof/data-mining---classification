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
import numpy as np
import pandas as pd
import KNN_classifier as knn
import cross_validation as cross
import beat_the_bench as beat
import matplotlib.pyplot as plt

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

def main():
	data = pd.read_csv('./datasets/train_set.csv', sep="\t")
	unseen_data = pd.read_csv('./datasets/test_set.csv', sep="\t")
	folds = 10
	file = open("../EvaluationMetric_10fold.csv",'w')
	file.write("Statistic Measure"+","+"Naive Bayes"+","+"Random Forest"+","+"SVM"+","+"KNN"+","+"My Method"+"\n")

	svd_model = TruncatedSVD(n_components=500, algorithm='randomized',n_iter=10, random_state=42)

	#add_stop_words = ["say","said","will","one","now","government","people","time","make","made","UK","US"]
	#enhanced_stop_words = ENGLISH_STOP_WORDS.union(add_stop_words)

	#KNN
	results_knn = cross.KNN_CrossValidation(folds, data)

	#Random Forest---accuracy=93.54%
 	text_clf5 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', RandomForestClassifier())])
 	results_rforest = cross.KfoldCrossValidation(text_clf5, folds, data)

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
 	text_clf1 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', svm.SVC(C=1.0, kernel='linear',gamma=0.001,probability=True))])
	results_svm = cross.KfoldCrossValidation(text_clf1, folds, data)
	
 	#Naive Bayes
 	text_clf4 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
 	results_bayes = cross.KfoldCrossValidation(text_clf4, folds, data)

 	#SGDClassifier
 	text_clf2 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None, n_jobs=-1))])
 	#cross.KfoldCrossValidation(text_clf2, folds, data)

 	#Neural Network
 	text_clf3 = Pipeline([('vect', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=7000)),('tfidf', TfidfTransformer()),('svd', svd_model),('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 10), random_state=1))])
 	#cross.KfoldCrossValidation(text_clf3, folds, data)

 	#My classifier
 	results_mymethod = cross.multi_classifier_KfoldCrossValidation(text_clf1, text_clf3, text_clf4, folds, data)

 	for i in range(4):
 		if(i==0):
 			file.write("Accuracy,");
 		elif(i==1):
 			file.write("Precision,");
 		elif(i==2):
 			file.write("Recall,");
 		elif(i==3):
 			file.write("F-Measure,");
 		file.write(results_bayes[i]+","+results_rforest[i]+","+results_svm[i]+","+results_knn[i]+","+results_mymethod[i]+"\n")
 	file.close()

 	#unseen_data_predict(text_clf, data, unseen_data)
 	beat.unseen_data_predict_combine_3classifiers(text_clf1,text_clf3,text_clf4,data,unseen_data)

if __name__ == "__main__":
	main()