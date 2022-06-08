#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 22:55:06 2022

@author: kosmobert
"""

import time
from os import listdir
from os import getcwd
from os import walk
from pathlib import Path

from multiprocessing import Pool
from multiprocessing import cpu_count
from joblib import dump
from pandas import read_excel
from pandas import concat
from pandas import to_datetime

# wrap your excel importer in a function that can be mapped
def Read_Excel(filename):
    'converts a filename to a pandas dataframe'
    return read_excel(filename)

def main():
    currentdir=Path(getcwd())
    libdir=currentdir.parent.absolute()
    years=next(walk(libdir))[1]
    start_time = time.time()
    pool = Pool(processes=cpu_count())

    files = []
    for year in years:
        filenames=listdir(str(libdir)+'/'+year+'/')
        files.extend([str(libdir) + '/' + year + '/' + filename for filename in filenames if (filename.endswith('.xlsx') and '~$' not in filename)])

    print()
    print('found', len(files), 'weeks with selections in the folder structure')
    print('will now load all files and concat to dataframe using all your cores at the same time')
    print('be patient for a moment until all selections are merged to one dataframe')
    print('it takes usually 1-3 seconds for each week, depending on your system')
    print('')

    df_list = pool.map(Read_Excel, files)
    df = concat(df_list, ignore_index=True)

    print('')
    print('look at the result, this is the merged raw dataframe, which is sorted according to selection time')
    print('--->')


    df['Created']=to_datetime(df['Created'])
    df=df.sort_values(by=['Created'])
    df = df.reset_index(drop=True)

    print(df)

    print("--- %s seconds ---" % (time.time() - start_time))

    dump(df, str(libdir)+'/mergedandsortedrawselctiondataframe.sav')

if __name__ == '__main__':
    main()
