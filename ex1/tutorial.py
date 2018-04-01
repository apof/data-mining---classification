from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd

train_data = pd.read_csv('./datasets/train_set.csv', sep="\t")
train_data = train_data[0:25]
print train_data.columns
print set(train_data['Category'])
print train_data
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
set(y)
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = count_vectorizer.fit_transform(train_data['Content'])
print X.shape
print X
print X.toarray()
print y

#clf = RandomForestClassifier()
clf = svm.SVC()

# fit train set
clf.fit(X, y)
# predict test set (here is the same as the train set)
y_pred = clf.predict(X)
dec = clf.decision_function(X)