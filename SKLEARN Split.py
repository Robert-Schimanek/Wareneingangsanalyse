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
#from sklearn.metrics import print
# Importiere von River
#from river import datasets
#from river import linear_model
#from river import metrics
#from river import compose
#from river import preprocessing
#from river import feature_extraction
#from river import stream
#from river.naive_bayes import MultinomialNB
#from river.linear_model import LogisticRegression
#from river.feature_extraction import BagOfWords,TFIDF
#from river.compose import Pipeline 


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


# Datensatz laden
rawdata = pd.read_csv('C:/Users/gezer/Desktop/AI.csv');    #print(rawdata.head(10));  #print(rawdata.dtypes)
# Datensatz aufteilen in Objekt- und Zahldaten
X = rawdata.select_dtypes(include=[object]).copy(); #print(X.head(3));  print(X.shape); print(X.columns)
Y = rawdata.select_dtypes(include=[np.number]).copy();  #print(Y.head(3));  print(Y.shape); print(Y.columns)

# LabelEncoder initialisieren
le = preprocessing.LabelEncoder()

# Mit LabelEncoder die Eingabedaten Transformieren
X2 = X.apply(le.fit_transform); #print(X2.head(5))

# Zahldatensatz und Transformiertes Datensatz zusammenfuegen
dataframe = pd.concat([X2,Y], axis=1);  #print(dataframe.head(10))

# Datensatz aufteilen in Eingabe und Augabedataframes
X = dataframe.drop('Real_PartNo', axis = 1);    #print(X)
Y = dataframe['Real_PartNo'];   #print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 48)

print("X_train shape:", X_train.shape); #print("X_train:", X_train)
print("Y_train shape:", Y_train.shape); #print("Y_train:", Y_train)
print("X_test shape:", X_test.shape);   #print("X_test:", X_test)
print("Y_test shape:", Y_test.shape);   #print("Y_test:", Y_test)

# StandardScale + LogisticRegression Modell Erstellen
#model = Pipeline([('SS', StandardScaler()), ('LR', LogisticRegression(max_iter=10000))])

# Modell Erstellen
model = HistGradientBoostingClassifier()

# Modell fuer die Eingabe und Augabe Datensaetze anpassen
model.fit(X_train, Y_train)
# Ergebniss des Modells ausgeben
print(model.score(X_test, Y_test))
print(model['LR'].coef_)

y_pred_prob=model.predict_proba(X_test)
predictions=model.predict(X_test)
print(predictions)

# Vorhersagen treffen
#print(accuracy_score(Y, Y_test, normalize=True, sample_weight=None))
print(model.predict([[10,11,30,2,0,1,6020008011,4047026041453]]))

sns.countplot(x=predictions, orient='h')
plt.show()
#print(predictions[:,0])
print(model['LR'].coef_)
print(model['LR'].intercept_)
print('Coefficients close to zero will contribute little to the end result')

num_err = np.sum(Y != model.predict(X))
print("Number of errors:", num_err)

def my_loss(y,w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w[i]
        s = s + (y_i_true - y_i_pred)**2
    return s

print("Loss:",my_loss(Y_test,predictions))

fpr, tpr, threshholds = roc_curve(Y_test,y_pred_prob[:,1])

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

accuracy=round(model['lr'].score(X_train, Y_train) * 100, 2)

print("Model Accuracy={accuracy}".format(accuracy=accuracy))

cm=confusion_matrix(Y_test,predictions)
print(cm)

#print(accuracy_score(Y, Y_test, normalize=True, sample_weight=None))