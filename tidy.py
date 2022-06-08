#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:24:03 2022

@author: kosmobert
"""

#os
from joblib import load, dump
import os
from pathlib import Path
import numpy as np
import pandas as pd
# search for rawdataname on your system
# specify the directory otherwise it will take long and it will just take
# the first one found on your system

rawdataname='mergedandsortedrawselctiondataframe.sav'
searchdir='/'
currentdir=Path(os.getcwd())
libdir=currentdir.parent.absolute()

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

path=find(rawdataname,searchdir)

dg=load(path)

dg.info(verbose=True, null_counts=True)

print('i remove now empty colums from the dataframe')
dg=dg.dropna(axis=1,how='all')

print('Did load the specified input data')
print('')
print('Look at the header of the Dataframe, its filled with features')
print('')
dg.info(verbose=True, null_counts=True)
print('')
print('The Dataframe has', len((list(dg))), 'feature columns/observations/measurements and a total of', dg.shape[0],'rows/selections' )
print('')
print('For tidying up the dataframe, i will now expand the column "Suggested OENs"')

split=dg['Suggested OENs'].str.split('/', 1, expand=True)
dg = dg.assign(Suggested_OENs_0=split[0])
n=1

while any(split[1].str.contains("/", case=False, na=False)):
    split=split[1].str.split('/', 1, expand=True)
    dg = dg.assign(first_name=split[0])
    name='Suggested_OENs_'+str(n)
    n=n+1
    dg=dg.rename(columns={"first_name": name})
    dg[name] = dg[name].replace(r'^\s*$', np.nan, regex=True)
    dg[name].fillna(value=np.nan, inplace=True)


print('')
print('Now the Suggested OENs are expanded')

dg=dg.drop(columns='Suggested OENs')

print('')
print('So i deleted the column with the Suggested OENs')
print('')
print('For tidying up the dataframe, i will now expand each core suggestion made by ceco statistics')

n=0
name='Suggested_OENs_'+str(n)
split=dg[name].str.split(';', 1, expand=True)

try:
    while any(split[1].str.contains(";", case=False, na=False)):

        name='Suggested_ID_'+str(n)
        dg = dg.assign(first_name=split[0])
        dg=dg.rename(columns={"first_name": name})
        dg[name] = dg[name].replace(r'^\s*$', np.nan, regex=True)
        dg[name].fillna(value=np.nan, inplace=True)

        split=split[1].str.split(';', 1, expand=True)
        name='Suggested_Product_Group_'+str(n)
        dg = dg.assign(first_name=split[0])
        dg=dg.rename(columns={"first_name": name})
        dg[name] = dg[name].replace(r'^\s*$', np.nan, regex=True)
        dg[name].fillna(value=np.nan, inplace=True)

        split=split[1].str.split(';', 1, expand=True)
        name='Suggested_OEN_'+str(n)
        dg = dg.assign(first_name=split[0])
        dg=dg.rename(columns={"first_name": name})
        dg[name] = dg[name].replace(r'^\s*$', np.nan, regex=True)
        dg[name].fillna(value=np.nan, inplace=True)

        n=n+1
        name='Suggested_OENs_'+str(n)
        split=dg[name].str.split(';', 1, expand=True)

except KeyError:
    n=1

print('')
print('Now the Suggested ID, product group, and OEN for each corea suggestion are tidy')

dg=dg.drop(columns=['Suggested_OENs_0','Suggested_OENs_1','Suggested_OENs_2','Suggested_OENs_3','Suggested_OENs_4'])

print('')
print('So i deleted the column with the Suggested OENs')


dg=dg.drop(columns=['ACTIVE_BOX_SEARCH_PARAMS'])


print('')
print('I dropped the column "ACTIVE_BOX_SEARCH_PARAMS" because the information is also in other columns of the dataframe')
print('')

print('Still, the column "Created" is not tidy. Lets have a look at it')
print('')

print(dg['Created'])
print('')

print('Lets create some cegorical data from ist... This takes some time... Get urself a coffe')
print('')

#1 Year the Selevtion was made
dg['Created:year'] = dg['Created'].dt.year
#2 Quarter of the year the selection was made
dg['Created:quarter'] = dg['Created'].dt.quarter
#3 Monthof the year the selection was made
dg['Created:month'] = dg['Created'].dt.month
#4 Week of the year the selection was made
dg['Created:week_of_year'] = dg['Created'].dt.isocalendar().week
#4 Day of the week the selection was made
dg['Created:day_of_week'] = dg['Created'].dt.day_of_week
#5 Seconds since beginning
dg['Created:begin'] = ( dg['Created'] - dg['Created'][0] ) / np.timedelta64(1, 's')
#6 Weeks since beginning
dg['Created:begin:weeks'] = ( dg['Created'] - dg['Created'][0] ) / np.timedelta64(1, 'W')


print('So the last colums contain year, month, week and day of year the selection was made')
print('')
dg.info(verbose=True, null_counts=True)
print('')

print('But this is not everything, we can gain more info from the created selection time')
print('')
print('In oder to use it we get rid of spaces in the ProductGroup EnteredReman and RealPartNo.')
print('')

dg['ProductGroup']=dg['Product Group'].str.replace(' ','')
dg['EnteredReman']=dg['Entered Reman'].str.replace(' ','')
dg['RealPartNo']=dg['Real PartNo'].str.replace(' ','')
dg['EffectiveReman']=dg['Effective Reman'].str.replace(' ','')

dg.info(verbose=True, null_counts=True)


dg['SuggestedOEN0']=dg['Suggested_OEN_0'].str.replace(' ','')
dg['SuggestedOEN0_truth']=dg['SuggestedOEN0']
dg['SuggestedOEN0_truth']=dg['RealPartNo']==dg['SuggestedOEN0']

dg['SuggestedOEN1']=dg['Suggested_OEN_1'].str.replace(' ','')
dg['SuggestedOEN1_truth']=dg['SuggestedOEN1']
dg['SuggestedOEN1_truth']=dg['RealPartNo']==dg['SuggestedOEN1']

dg['SuggestedOEN2']=dg['Suggested_OEN_2'].str.replace(' ','')
dg['SuggestedOEN2_truth']=dg['SuggestedOEN2']
dg['SuggestedOEN2_truth']=dg['RealPartNo']==dg['SuggestedOEN2']

dg['SuggestedOEN3']=dg['Suggested_OEN_3'].str.replace(' ','')
dg['SuggestedOEN3_truth']=dg['SuggestedOEN3']
dg['SuggestedOEN3_truth']=dg['RealPartNo']==dg['SuggestedOEN3']

dg['SuggestedOEN4']=dg['Suggested_OEN_4'].str.replace(' ','')
dg['SuggestedOEN4_truth']=dg['SuggestedOEN4']
dg['SuggestedOEN4_truth']=dg['RealPartNo']==dg['SuggestedOEN4']

dg.info(verbose=True, null_counts=True)


print('There are some RealPartNo that are not really RealPartNo')
print('This includes for example Bosch parts with the initals 0 986')
print('This includes for example  parts with the initals KS01000')

print('I will create a tidy RealPartNo called RealPartNoTidy')


dg['RealPartNoTidy']=dg['RealPartNo']
dg.loc[ dg['RealPartNo'].str.startswith('0986',na=False), 'RealPartNoTidy']=np.nan
dg.loc[ dg['RealPartNo'].str.startswith('KS01000',na=False), 'RealPartNoTidy']=np.nan

dg.info(verbose=True, null_counts=True)

dg['SuggestedOEN0']=dg['Suggested_OEN_0'].str.replace(' ','')
dg['SuggestedOEN0_truth_verfied']=dg['SuggestedOEN0']
dg['SuggestedOEN0_truth_verfied']=dg['RealPartNoTidy']==dg['SuggestedOEN0']

dg['SuggestedOEN1']=dg['Suggested_OEN_1'].str.replace(' ','')
dg['SuggestedOEN1_truth_verfied']=dg['SuggestedOEN1']
dg['SuggestedOEN1_truth_verfied']=dg['RealPartNoTidy']==dg['SuggestedOEN1']

dg['SuggestedOEN2']=dg['Suggested_OEN_2'].str.replace(' ','')
dg['SuggestedOEN2_truth_verfied']=dg['SuggestedOEN2']
dg['SuggestedOEN2_truth_verfied']=dg['RealPartNoTidy']==dg['SuggestedOEN2']

dg['SuggestedOEN3']=dg['Suggested_OEN_3'].str.replace(' ','')
dg['SuggestedOEN3_truth_verfied']=dg['SuggestedOEN3']
dg['SuggestedOEN3_truth_verfied']=dg['RealPartNoTidy']==dg['SuggestedOEN3']

dg['SuggestedOEN4']=dg['Suggested_OEN_4'].str.replace(' ','')
dg['SuggestedOEN4_truth_verfied']=dg['SuggestedOEN4']
dg['SuggestedOEN4_truth_verfied']=dg['RealPartNoTidy']==dg['SuggestedOEN4']


# Create RealIdent from RealPartNo
dg['RealIdentNo'] = dg['RealPartNoTidy']

dg.info(verbose=True, null_counts=True)

print('Then we set the IDs / classes our algorithms should look for.')
print('')

# For some Product Groups fill RealIdent with Entered Reman or 'DieselInjector' or 'UnitInjector' or 'CommonRailInjector',
dg.loc[ dg['ProductGroup'] == 'BrakeCaliper' , 'RealIdentNo'] = dg[dg['ProductGroup'].str.contains( 'BrakeCaliper' )==True]['EnteredReman']
dg.loc[ dg['ProductGroup'] == 'DieselInjector' , 'RealIdentNo'] = dg[dg['ProductGroup'].str.contains( 'DieselInjector' )==True]['EnteredReman']
dg.loc[ dg['ProductGroup'] == 'UnitInjector' , 'RealIdentNo'] = dg[dg['ProductGroup'].str.contains( 'UnitInjector' )==True]['EnteredReman']
dg.loc[ dg['ProductGroup'] == 'CommonRailInjector' , 'RealIdentNo'] = dg[dg['ProductGroup'].str.contains( 'CommonRailInjector' )==True]['EnteredReman']

print('Finally we calculate the time between the selections based on their ids. This takes a lot of time. For this two coffes should suffice.')
print('')

dg['Created:diff']=dg['Created'].diff().fillna(pd.Timedelta(days=0))/np.timedelta64(1, 's')

for x in dg['RealIdentNo'].unique():
    dg.loc[ dg['RealIdentNo'] == x, 'Created:diff:RealIdentNo']=dg.loc[ dg['RealIdentNo'] == x, 'Created'].diff()
dg['Created:diff:RealIdentNo']=dg['Created:diff:RealIdentNo'].fillna(pd.Timedelta(days=0))
dg['Created:diff:RealIdentNo']=dg['Created:diff:RealIdentNo'] / np.timedelta64(1, 'D')
dg['Created:diff:RealIdentNo']=round(dg['Created:diff:RealIdentNo'])

for x in dg['Bar Code'].unique():
    dg.loc[ dg['Bar Code'] == x, 'Created:diff:Bar Code']=dg.loc[ dg['Bar Code'] == x, 'Created'].diff()
dg['Created:diff:Bar Code']=dg['Created:diff:Bar Code'].fillna(pd.Timedelta(days=0))
dg['Created:diff:Bar Code']=dg['Created:diff:Bar Code'] / np.timedelta64(1, 'D')
dg['Created:diff:Bar Code']=round(dg['Created:diff:Bar Code'])

for x in dg['Customer Number'].unique():
    dg.loc[ dg['Customer Number'] == x, 'Created:diff:Customer Number']=dg.loc[ dg['Customer Number'] == x, 'Created'].diff()
dg['Created:diff:Customer Number']=dg['Created:diff:Customer Number'].fillna(pd.Timedelta(days=0))
dg['Created:diff:Customer Number']=dg['Created:diff:Customer Number'] / np.timedelta64(1, 'D')
dg['Created:diff:Customer Number']=round(dg['Created:diff:Customer Number'])

for x in dg['Customer Delv No.'].unique():
    dg.loc[ dg['Customer Delv No.'] == x, 'Created:diff:Customer Delv No.']=dg.loc[ dg['Customer Delv No.'] == x, 'Created'].diff()
dg['Created:diff:Customer Delv No.']=dg['Created:diff:Customer Delv No.'].fillna(pd.Timedelta(days=0))
dg['Created:diff:Customer Delv No.']=dg['Created:diff:Customer Delv No.'] / np.timedelta64(1, 'D')
dg['Created:diff:Customer Delv No.']=round(dg['Created:diff:Customer Delv No.'])

for x in dg['ProductGroup'].unique():
    dg.loc[ dg['ProductGroup'] == x, 'Created:diff:ProductGroup']=dg.loc[ dg['ProductGroup'] == x, 'Created'].diff()
dg['Created:diff:ProductGroup']=dg['Created:diff:ProductGroup'].fillna(pd.Timedelta(days=0))
dg['Created:diff:ProductGroup']=dg['Created:diff:ProductGroup'] / np.timedelta64(1, 'D')
dg['Created:diff:ProductGroup']=round(dg['Created:diff:ProductGroup'])


print('Technically we are tidy now, but ill leave the created time in the frame for your transformations ')
print('')

dg.info(verbose=True, null_counts=True)


dump(dg, str(libdir)+'/tidydataframe.sav')
