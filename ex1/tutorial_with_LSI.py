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
import pandas as pd

data = pd.read_csv('./datasets/train_set.csv', sep="\t")
data = data[0:10]
# print data[0:10]
# X_train, X_test, y_train, y_test = train_test_split(data, data['Category'],test_size=0.1,random_state=0)
# print X_train.shape
# print X_test.shape

kf = KFold(n_splits=10)
for train, test in kf.split(data):
	print train.shape
	print test.shape
	print("%s %s" % (train, test))
	print data['Category']

# train_data = data[0:10000]
# print train_data.columns
# print set(train_data['Category'])
# print train_data
# le = preprocessing.LabelEncoder()
# le.fit(train_data["Category"])
# y = le.transform(train_data["Category"])
# set(y)

# vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,max_features=1000)
# XX = vectorizer.fit_transform(train_data['Content'])
# transformer = TfidfTransformer(use_idf=False).fit(XX)
# X = transformer.fit_transform(XX)

# n_comp = 150
# svd = TruncatedSVD(n_components=n_comp)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)

# X = lsa.fit_transform(X)

# print X.shape
# print X
# print y

# clf = RandomForestClassifier()

# # fit train set
# clf.fit(X, y)

# # predict test set (here is the same as the train set)
# train_data = data[10000:11000]
# le.fit(train_data["Category"])
# y = le.transform(train_data["Category"])
# set(y)
# X2 = vectorizer.fit_transform(train_data['Content'])
# svd = TruncatedSVD(n_components=n_comp)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
# X2 = lsa.fit_transform(X2)

# y_pred = clf.predict(X2)
# print classification_report(y, y_pred, target_names=list(le.classes_))
