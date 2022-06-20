# Importiere von skmultiflow
from tkinter import S
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier, HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestRegressor, LeveragingBaggingClassifier
from skmultiflow.data import DataStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import KNNClassifier
from skmultiflow.evaluation import EvaluatePrequential
# Importiere von Sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Importiere von Andere
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
import time
import warnings

# Zeit starten und warnings ausschalten weil alte Funktionen genutzt werden
start_time = time.time()
warnings.filterwarnings("ignore")

# Datensatz laden
datasetX = pd.read_csv('DatasetX.csv', low_memory=False); #print(datasetX.head(10));  print(datasetX.dtypes)
datasetY = pd.read_csv('DatasetY.csv', low_memory=False); #print(datasetY.head(10));  print(datasetY.dtypes)

# Encoder initialisieren
OE = preprocessing.OrdinalEncoder()
LE = preprocessing.LabelEncoder()

# Datensatz Encoden
datasetX = OE.fit_transform(datasetX)
datasetY = datasetY.apply(LE.fit_transform)

# Datensatz in Dataframe umwandeln
dfX = pd.DataFrame(datasetX)
dfY = pd.DataFrame(datasetY)

# Ein und Ausgabewerte zusammenfassen
dataframe = pd.concat([dfX,dfY], axis=1)
dataframe.to_csv('neu.csv', index = 0)

# Datenframe in ein Stream umwandeln
stream=DataStream(dataframe, target_idx=7)

# Modelle importieren
EFDT = ExtremelyFastDecisionTreeClassifier()
HAT = HoeffdingAdaptiveTreeClassifier()
HT = HoeffdingTreeClassifier()
NB = NaiveBayes()
FR = AdaptiveRandomForestRegressor()
LB = LeveragingBaggingClassifier()
KNN = KNNClassifier()
CNB = CategoricalNB()
SGD = SGDClassifier()
BNB = BernoulliNB()
MNB = MultinomialNB()
GNB = GaussianNB()
CLF = tree.DecisionTreeClassifier()
SRFC = StreamingRFC()

model = SRFC
model_names = ['DTC']

# BATCHLEARNING (IN ECHT AUCH INKREMENTELL ABER PSCHT)
#-------------------------------------------------------------------------------------------------------------------------------------------
print('Batchlearning Modell: ', model_names)
model.fit(dfX, dfY.values.ravel())
print('Genauigkeit:', model.score(dfX,dfY))
#print(model.predict([[0, 0, 0, 0, 0, 0, 0]]))
print('--------------------')
#-------------------------------------------------------------------------------------------------------------------------------------------

#'''
# INCREMENTAL LEARNING
#-------------------------------------------------------------------------------------------------------------------------------------------
# Variablen fuer die Iteration
k=0
IL_samples = 0
IL_correct_cnt = 0
IL_max_samples = 65000
IL_pred = np.empty([IL_max_samples], dtype=int)
IL = np.empty([IL_max_samples], dtype=int)

# Inkrementell existierendes Modell trainieren und testen
while IL_samples < IL_max_samples and stream.has_more_samples():
    X, y = stream.next_sample();                #print('IL:',y)
    y_pred = model.predict(X);                  #print(y_pred)
    if y[0] == y_pred[0]:
        IL_correct_cnt += 1                     # Anzahl der richtigen Schaetzungen
    if k % 500 == 0: 
        print(IL_samples, 'samples tested')
    model.partial_fit(X, y)
    IL_pred[IL_samples] = y_pred                # Schaetzung Speichern
    IL[IL_samples] = y                          #Anzahl der Iterationen
    IL_samples += 1;                            
    k = k + 1 
#-------------------------------------------------------------------------------------------------------------------------------------------


# Ergebnisse ausgeben
#-------------------------------------------------------------------------------------------------------------------------------------------
print('Ergebnisse IL Modell: ', model_names)
num_err = np.sum(y != model.predict(X))
print('--------------------')
print('{} samples tested.'.format(IL_samples))                                      # IL samples analisiert
print("Number of errors:", num_err)                                                 # Anzahl der Fehler
print('--------------------')
print('Median Accuracy IL testing: {}'.format(IL_correct_cnt / IL_samples))         # IL Genauigkeit
c_pred = Counter(IL_pred);                                                          #print(c_pred);      #print(IL_pred)
c_real = Counter(IL);                                                               #print(c_real);      #print(IL)
print("--- %s seconds ---" % (time.time() - start_time));                           # Dauer der Analyse
#-------------------------------------------------------------------------------------------------------------------------------------------


# IL Plotten
#-------------------------------------------------------------------------------------------------------------------------------------------
# Create dictionaries from lists with this format: 'letter':count
dict1 = dict(zip(*np.unique(IL_pred, return_counts=True)))
dict2 = dict(zip(*np.unique(IL, return_counts=True)))

# Add missing letters with count=0 to each dictionary so that keys in
# each dictionary are identical
only_in_set1 = set(dict1)-set(dict2)
only_in_set2 = set(dict2)-set(dict1)
dict1.update(dict(zip(only_in_set2, [0]*len(only_in_set2))))
dict2.update(dict(zip(only_in_set1, [0]*len(only_in_set1))))

# Sort dictionaries alphabetically
dict1 = dict(sorted(dict1.items()))
dict2 = dict(sorted(dict2.items()))

# Create grouped bar chart
xticks = np.arange(len(dict1))
bar_width = 0.3
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(xticks-bar_width/2, dict1.values(), bar_width,
       color='blue', alpha=0.5, label='Predicted No.')
ax.bar(xticks+bar_width/2, dict2.values(), bar_width,
       color='red', alpha=0.5, label='Actual No.')

# Set annotations, x-axis ticks and tick labels
ax.set_ylabel('Counts')
ax.set_title('Real PartNo: Testing analysis')
ax.set_xticks(xticks)
ax.set_xticklabels(dict1.keys())
ax.legend(frameon=False)
plt.show()
#'''
#-------------------------------------------------------------------------------------------------------------------------------------------


# Methode 2 (gut fuer mehrere Modelle)
#-------------------------------------------------------------------------------------------------------------------------------------------
'''
eval = EvaluatePrequential(show_plot=True,
                           max_samples= 65000,
                           metrics=['accuracy', 'running_time', 'model_size', 'true_vs_predicted'],
                           output_file='output.csv',
                           n_wait=500,
                           pretrain_size=0,
                           data_points_for_classification=True
                           )

eval.evaluate(stream=stream, model= model, model_names=model_names)
'''
#-------------------------------------------------------------------------------------------------------------------------------------------


# Datensatz invertieren
#-------------------------------------------------------------------------------------------------------------------------------------------
'''
X = i;  X_pred = i_pred;    #print(X)
inv = LE.inverse_transform(X)
inv_pred = LE.inverse_transform(X_pred)
inv = pd.DataFrame(inv)
inv_pred = pd.DataFrame(inv_pred)
inv.rename(columns={0: 'Real PartNo'}, inplace=True)
inv_pred.rename(columns={0: 'pred. Real PartNo'}, inplace=True)
invNo = pd.concat([inv_pred,inv], axis=1)
invNo.to_csv('C:/Users/gezer/Desktop/results.csv', index = 0); #print(inv)
#inv_pred.to_csv('C:/Users/gezer/Desktop/inv_pred.csv', index = 0); #print(inv_pred)
'''
#-------------------------------------------------------------------------------------------------------------------------------------------


# Stream als csv ausgeben
#-------------------------------------------------------------------------------------------------------------------------------------------
'''
stream = DataStream(dataframe, target_idx=7, allow_nan=True)
X, y = stream.next_sample(5); 
df = pd.DataFrame(np.hstack((X, y.reshape(-1,1))),
                  columns=['attr_{}'.format(i) for i in range(X.shape[1])] + ['target'])
df.target = df.target.astype(int)
df.to_csv('C:/Users/gezer/Desktop/streamlol.csv')
'''
#-------------------------------------------------------------------------------------------------------------------------------------------
