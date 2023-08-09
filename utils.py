import pandas as pd 
import numpy as np 
import os
import time

__all__ =['import_dataset',
          'read_data_list',
          'timer']

def import_dataset(path : str) -> pd.DataFrame:
    '''
    path + file name
    ### Example
    >>> path = './{DATASET NAME.EXT}'
    >>> import_dataset(path)
    >>> print(data.head(3))
    ...       0     1     2    3     4     5     6 Class
    ... 0  0.49  0.29  0.48  0.5  0.56  0.24  0.35     0
    ... 1  0.07  0.40  0.48  0.5  0.54  0.35  0.44     0
    ... 2  0.56  0.40  0.48  0.5  0.49  0.37  0.46     1
    '''
    path = path
    # dataset = pd.read_csv(path, header = 0, comment = '@', sep = ', ', engine = 'python')
    dataset = pd.read_csv(path, header = None, comment = '@', engine = 'python')


    name = [str(x) for x in np.arange(dataset.shape[1] - 1)]
    name.append('Class')
    # dataset = dataset.rename(columns = name, inplace = False)
    dataset.columns = name
    # dataset['Class'] = dataset['Class'].astype('string')
    # dataset['Class'] = dataset['Class'].str.strip()
    
    # dataset['Class'].replace(['negative', ' negative', 'negative '], '0', inplace = True)
    # dataset['Class'].replace(['positive', ' positive', 'positive '], '1', inplace = True)
    for i in range(dataset.shape[0]):
        if (dataset.iloc[i, -1] == ' negative') or (dataset.iloc[i, -1] == 'negative'):
            dataset.iloc[i, -1] = 0
        else:
            dataset.iloc[i, -1] = 1

    dataset['Class'] = dataset['Class'].astype("category")
    del name
    return dataset

def read_data_list(path : str = None, fileExt : str = '.dat') -> list:

    temp = os.listdir(path = path)
    data_list = []

    for i in temp:
        a = os.path.splitext(i)
        if a[-1] == fileExt:
            data_list.append(a[-2])
            
    del a, temp        
    return data_list

def timer(func):
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        gap = time.time() - start
        out = [time.asctime(), int(gap  // 60), gap %60]
        print('Process has been compeleted! \nSys time : {0:s} Exe time : {1:3d} mins {2:.4f} s'.format(*out))
        # logging.debug('Process has been compeleted! \n Exe time : {0:3d} mins {1:.4f} s'.format(*out))
        return result
    return wrapper


