# Importiere Bibliotheken
from joblib import load
import numpy as np
import pandas as pd


#CSV Dateien einlesen
dataset = load('C:/Users/gezer/Desktop/Python/tidydataframe.sav')
df = pd.DataFrame(dataset)
newcolumn = pd.DataFrame(dataset)
export = pd.read_csv('C:/Users/gezer/Desktop/export.csv')
filter = pd.read_csv('C:/Users/gezer/Desktop/filter list.csv')
print(df); print(newcolumn)

# Leerzeichen von Real PartNo entfernen
df['Real PartNo'] = df['Real PartNo'].str.replace(" ","")
newcolumn['Real PartNo'] = newcolumn['Real PartNo'].str.replace(" ","")

# Unwichtige Spalten loeschen und neue spalte importieren
df.drop(df.columns.difference(['Customer Number', 'Real PartNo', 'Delivery ID', 'Customer Delv No.', 'Product Group', 'Bar Code', 'Box exists', 'Box code scanable']), axis=1, inplace=True)
newcolumn.drop(newcolumn.columns.difference(['Box code', 'Box number selected', 'CBN', 'Effective Reman']), axis=1, inplace=True)
print(df);  print(newcolumn)

# Datensatz mit neue Spalte zusammenfuegen
dfnew = pd.concat([df,newcolumn], axis=1)

# Spalten mit leeren Inhalten loeschen
dfnew.replace('', np.nan, inplace=True)
dfnew.dropna(inplace=True)
print(dfnew)


# Datein nach Real PartNo filtern
#'''
list = filter['Real PartNo'].to_list();   #print(list)
for i in list:
    filtered = dfnew.loc[(dfnew['Real PartNo'] == i)]; #print(k)
    export = pd.concat([export,filtered])
export.drop(export.index[[0]], inplace=True)
print(export)
#'''

# Duplikate entfernen
#export = export.drop_duplicates(keep=False)

# Datensatz durchmischen
export = export.sample(frac=1)

# Datensatz in X und Y Anteile aufteilen
exportX = export.drop('Real PartNo', axis=1)
exportY = export.drop(export.columns.difference(['Real PartNo']), axis=1)

# Datensatz ausgeben 
exportX.to_csv('C:/Users/gezer/Desktop/DatasetX.csv', index=False)
exportY.to_csv('C:/Users/gezer/Desktop/DatasetY.csv', index=False)

'''
# Real PartNo zaehlen
dups = filtered.pivot_table(columns=['Real PartNo'], aggfunc='size')
dups.to_csv('C:/Users/gezer/Desktop/duplicates.csv', index=True)
'''