# Importiere Bibliotheken
from joblib import load
import numpy as np
import pandas as pd

#TODO: Datensatz muss noch nach Datum sortiert werden und danach mischen entfernen

#CSV Dateien einlesen
dataset = load('C:/Users/gezer/Desktop/Wareneingangsanalyse/tidydataframe.sav')
export = pd.read_csv('https://raw.githubusercontent.com/Tenny131/Wareneingangsanalyse/main/export.csv')
filter = pd.read_csv('https://raw.githubusercontent.com/Tenny131/Wareneingangsanalyse/main/filter%20list.csv')
df = pd.DataFrame(dataset)
# Leerzeichen von Real PartNo entfernen
df['Real PartNo'] = df['Real PartNo'].str.replace(" ","")
#df['Created'] =pd.to_datetime(df.Created)
# Unwichtige Spalten loeschen und neue spalte importieren
df.drop(df.columns.difference(['Customer Number', 'Real PartNo', 'Delivery ID', 'Customer Delv No.', 'Product Group', 'Bar Code', 'Box exists', 'Box code scanable', 'Box code', 'Box number selected', 'CBN', 'Effective Reman']), axis=1, inplace=True)
print(df)

# Zeilen mit leeren Real PartNo loeschen und leere Zeilen bei anderen Spalten mit 0 fuellen
#df.fillna(0, inplace=True)
#df['Real PartNo'].replace(0, np.nan, inplace = True)
df.replace('', np.nan, inplace=True)
df.dropna(inplace=True)
print(df)

# Datein nach Real PartNo filtern <100 werden geloscht
list = filter['Real PartNo'].to_list();   #print(list)
for i in list:
    filtered = df.loc[(df['Real PartNo'] == i)]; #print(k)
    export = pd.concat([export,filtered])
export.drop(export.index[[0]], inplace=True)
print(export)

# Datensatz nach Datum sortieren
#export = export.sort_values(by='Created')
#export.drop(['Created'],axis=1, inplace=True)

export = export.sample(frac=1)

# Datensatz in X und Y Anteile aufteilen
exportX = export.drop('Real PartNo', axis=1)
exportY = export.drop(export.columns.difference(['Real PartNo']), axis=1)
print(exportX)
print(exportY)
# Datensatz ausgeben 
exportX.to_csv('C:/Users/gezer/Desktop/Wareneingangsanalyse/DatasetX.csv', index=False)
exportY.to_csv('C:/Users/gezer/Desktop/Wareneingangsanalyse/DatasetY.csv', index=False)

# Real PartNo zaehlen
dups =exportY['Real PartNo'].value_counts()
dups.to_csv('C:/Users/gezer/Desktop/Wareneingangsanalyse/duplicates.csv', index=True)
print(dups)