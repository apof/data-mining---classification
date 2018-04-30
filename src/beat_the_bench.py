import numpy as np
from operator import itemgetter

#This function uses 3 classifiers to predict the label of unseen data
#It also uses the title efficiently
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