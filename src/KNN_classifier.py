from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from numpy import linalg as LA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import math
from operator import itemgetter




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