from sklearn.model_selection import KFold
import numpy as np
from sklearn import metrics
import KNN_classifier as knn
from operator import itemgetter


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
		precision += metrics.precision_score(data['Category'][test_indices], predicted, average='macro')
		recall += metrics.recall_score(data['Category'][test_indices], predicted, average='macro')
		fmeasure += metrics.f1_score(data['Category'][test_indices], predicted, average='macro')

		print str(count) + "st fold completed"
		print metrics.accuracy_score(data['Category'][test_indices], predicted)
		print "F-Measure = " + str(fmeasure)
		count += 1

	print "Precision = " + str(precision/folds)
	print "Accuracy = " + str(accuracy/folds)
	print "Recall = " + str(recall/folds)
	print "F-Measure = " + str(fmeasure/folds)

	results = []
	results.append(str(accuracy/folds))
	results.append(str(precision/folds))
	results.append(str(recall/folds))
	results.append(str(fmeasure/folds))
	return results

def KNN_CrossValidation(folds, data):
	k_fold = KFold(n_splits=folds)
	accuracy = 0
	precision = 0
	recall = 0
	fmeasure = 0
	count = 1
	for train_indices, test_indices in k_fold.split(data['Category']):
		#text_clf.fit(data['Content'][train_indices], data['Category'][train_indices])

		predicted = knn.KNN(data['Content'][train_indices], data['Category'][train_indices], data['Content'][test_indices])
		predicted = np.asarray(predicted)
		accuracy += metrics.accuracy_score(data['Category'][test_indices], predicted) 
		precision += metrics.precision_score(data['Category'][test_indices], predicted, average='macro')
		recall += metrics.recall_score(data['Category'][test_indices], predicted, average='macro')
		fmeasure += metrics.f1_score(data['Category'][test_indices], predicted, average='macro')

		print str(count) + "st fold completed"
		count += 1

	print "Precision = " + str(precision/folds)
	print "Accuracy = " + str(accuracy/folds)
	print "Recall = " + str(recall/folds)
	print "F-Measure = " + str(fmeasure/folds)

	results = []
	results.append(str(accuracy/folds))
	results.append(str(precision/folds))
	results.append(str(recall/folds))
	results.append(str(fmeasure/folds))	
	return results

def multi_classifier_KfoldCrossValidation(text_clf1, text_clf2, text_clf3, folds, data):
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
		
		text_clf1.fit(data['Content'][train_indices], data['Category'][train_indices])
		predicted1 = text_clf1.predict(data['Content'][test_indices])
		y = text_clf1.predict_proba(data['Content'][test_indices])
		text_clf1.fit(data['Title'][train_indices], data['Category'][train_indices])
		y1 = text_clf1.predict_proba(data['Title'][test_indices])
		for i in range(len(y)):
			mm = max(y[i]+y1[i])
			index = np.where((y[i]+y1[i]) == mm)
			predicted1[i] = category_map[index[0][0]]

		text_clf2.fit(data['Content'][train_indices], data['Category'][train_indices])
		predicted2 = text_clf2.predict(data['Content'][test_indices])
		y = text_clf2.predict_proba(data['Content'][test_indices])
		text_clf2.fit(data['Title'][train_indices], data['Category'][train_indices])
		y1 = text_clf2.predict_proba(data['Title'][test_indices])
		for i in range(len(y)):
			mm = max(y[i]+y1[i])
			index = np.where((y[i]+y1[i]) == mm)
			predicted2[i] = category_map[index[0][0]]

		text_clf3.fit(data['Content'][train_indices], data['Category'][train_indices])
		predicted3 = text_clf3.predict(data['Content'][test_indices])
		y = text_clf3.predict_proba(data['Content'][test_indices])
		text_clf3.fit(data['Title'][train_indices], data['Category'][train_indices])
		y1 = text_clf3.predict_proba(data['Title'][test_indices])
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
				predicted1[j] = sorted_hashmap[0][0]

		accuracy += metrics.accuracy_score(data['Category'][test_indices], predicted1)  
		precision += metrics.precision_score(data['Category'][test_indices], predicted1, average='macro')
		recall += metrics.recall_score(data['Category'][test_indices], predicted1, average='macro')
		fmeasure += metrics.f1_score(data['Category'][test_indices], predicted1, average='macro')

		print str(count) + "st fold completed"
		print "Precision = " + str(precision)
		print "Accuracy = " + str(accuracy)
		print "Recall = " + str(recall)
		print "F-Measure = " + str(fmeasure)
		count += 1

	print "Precision = " + str(precision/folds)
	print "Accuracy = " + str(accuracy/folds)
	print "Recall = " + str(recall/folds)
	print "F-Measure = " + str(fmeasure/folds)

	results = []
	results.append(str(accuracy/folds))
	results.append(str(precision/folds))
	results.append(str(recall/folds))
	results.append(str(fmeasure/folds))

	return results