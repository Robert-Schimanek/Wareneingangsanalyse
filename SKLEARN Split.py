from glob import glob
import itertools
import os.path
import re
import tarfile
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from html.parser import HTMLParser
from urllib.request import urlretrieve
from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from skmultiflow.data import DataStream
from sklearn import preprocessing
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingClassifier

# Importiere von Sklearn
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Importiere von Rest
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as fpr
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import CategoricalNB


X = pd.read_csv('C:/Users/gezer/Desktop/Wareneingangsanalyse/DatasetX.csv', low_memory=False); #print(datasetX.head(10));  print(datasetX.dtypes)
y = pd.read_csv('C:/Users/gezer/Desktop/Wareneingangsanalyse/DatasetY.csv', low_memory=False); #print(datasetY.head(10));  print(datasetY.dtypes)

# Encoder initialisieren
OE = preprocessing.OrdinalEncoder()
LE = preprocessing.LabelEncoder()

# Datensatz Encoden
datasetX = OE.fit_transform(X)
datasetY = y.apply(LE.fit_transform)
#datasetX2 = OE.fit_transform(datasetX2)
#datasetY2 = datasetY2.apply(LE.fit_transform)

# Datensatz in Dataframe umwandeln
X = pd.DataFrame(datasetX)
y = pd.DataFrame(datasetY)

#X, y = make_classification(n_features=4, random_state=0, n_classes=2)
clf = CategoricalNB()
clf.fit(X, y.values.ravel())

print(X)
print(y)

print(clf.score(X,y))
print(clf.predict([[0, 0, 0, 0, 0, 0, 0]]))


'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 48)

# Modell Erstellen
model = HistGradientBoostingClassifier()

# Modell fuer die Eingabe und Augabe Datensaetze anpassen
model.fit(X_train, Y_train)
# Ergebniss des Modells ausgeben
print(model.score(X_train, Y_train))

y_pred_prob=model.predict_proba(X_test)
predictions=model.predict(X_test)
print(predictions)

# Vorhersagen treffen
print(accuracy_score(Y, Y_test, normalize=True, sample_weight=None))
print(model.predict([[10,11,30,2,0,1,6020008011,4047026041453]]))

accuracy=round(model['lr'].score(X_train, Y_train) * 100, 2)

print("Model Accuracy={accuracy}".format(accuracy=accuracy))
'''