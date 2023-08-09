### Necessary library
#general_frame_work
import pandas as pd
import numpy as np
#evaluation
from .utils import timer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import  confusion_matrix, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from imblearn.metrics import geometric_mean_score
#store model
import pickle
from typing import Literal


class Train_model():
    """
    # Parameters
    train_data : pd.DataFrame, default = None
        The datasets that used to buildi a classification model and validate.
        NOTE: The target variable (or Y) of a given dataset should be placed at the last column.

    classifier : any, default = None
        A sklearn classifier that accepts the array-liked dataset.

    parameters_dict : dict, default = None
        A dictionary-liked parameter that used to tune the parameters of the given classifier.

    resampler : any, default = None
        A resampling method that is capable to fit a array-liked dataset

    scaling : Literal, default = None
        MinMaxScaler ('MinMax'), StandardScaler('Std) for scaling the data. If None, do not scaling.

    kfold : int, default = 5
        The number of folds that datset will be seperated in validation.
    ---
    # Artributes
    evaluation_outcome : pd.DataFrame
        The average value of stratified k-fold validation.

    best_para : dict
        The best set of parameters of classifier in the given parameter dict (or self.parameters_dict).
    ---
    # Example
    >>> import pandas as pd
    >>> import smote_variants as sv
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from validation_module import Train_model
    >>> 
    >>> data = pd.DataFrame([[1, 2, 1], [1, 0, 1], [10, 4, 0], [10, 0, 0], [10, 2, 0], [1, 4, 1], [10, 4, 0], [10, 4, 0], [10, 4, 0], [10, 4, 0]])

    // Suppose the last column is the label of test dataset.
    >>> X, Y = data.iloc[:, 0:-1], data.iloc[:, -1]
    >>> Y.value_counts()
    >>> 0.0    7
        1.0    3

    >>> classifier = DecisionTreeClassifier()
    >>> resampler = sv.ROSE()
    >>> 
    >>> train_model = Train_model(
        train_data = data, 
        classifier = classifier,
        resampler = resampler
        kfold = 2)
    >>> 
    >>> train_model.evaluation()
    >>> print(train_model.evaluation_outcome)
           Accuracy  AUC  F1-score  Recall  Precision  G-mean
        0       1.0  1.0       1.0     1.0        1.0     1.0
    """

    def __init__(
        self,
        train_data: pd.DataFrame = None,
        classifier: any = None,
        parameters_dict: dict = None,
        resampler: any = None,
        scaling: Literal['MinMax', 'Std', None] = None,
        kfold: int = 5,
         ):
        # Parameters
        self.train_data : pd.DataFrame = train_data
        self.kfold : int = kfold
        self.parameters_dict = parameters_dict
        self.scaling : str = scaling
        self.resampler = resampler
        self.classifier = classifier
        # Attributes
        self.evaluation_outcome : pd.DataFrame = None
        self.best_para :dict = None
            

    def get_preprocessed_data(self, data: pd.DataFrame = None):
        '''
        Given an imbalanced dataset and resampler, then return a processed balanced dataset.
        '''
        
        if self.resampler == None:
            data = data.reset_index(drop = True)
            return data.iloc[:, 0:-1], data.iloc[:, -1]
        
        # If the resampler is a k-NN based algorithm, and if the number of minority class instances is less that default k(5)
        # , the k would be set to the number of minority class instances in the given dataset.
        # if 'n_neighbors' in dir(self.resampler) and (min(data.Class.value_counts()[:]) < self.resampler.get_params()['n_neighbors']):
        if 'n_neighbors' in dir(self.resampler) and (min(data.Class.value_counts()[:]) < self.resampler.n_neighbors):
            self.resampler.set_params(**{'n_neighbors': min(data.Class.value_counts()[:]) - 1})
        
        # if 'k_neighbors' in dir(self.resampler) and (min(data.Class.value_counts()[:]) < self.resampler.get_params()['k_neighbors']):
        if 'k_neighbors' in dir(self.resampler) and (min(data.Class.value_counts()[:]) < self.resampler.k_neighbors):
            self.resampler.set_params(**{'k_neighbors': min(data.Class.value_counts()[:]) - 1})

        b_X, b_Y = self.resampler.fit_resample(np.array(data.iloc[:, 0:-1]), np.array(data.iloc[:, -1]))
        
        b_X = pd.DataFrame(b_X, columns = [str(x) for x in range(b_X.shape[1])])
        b_Y = pd.Series(b_Y, name = 'Class')
        del data
        return b_X, b_Y

    def tune_para(self):
        '''Given classifier and dict-liked parameters seed return 
        the best parameter settings of classifier.
        '''
        grid_search = GridSearchCV(
            estimator = self.classifier,
            param_grid = self.parameters_dict,
            cv = self.kfold,
            n_jobs = -1,
            scoring = 'roc_auc'
            )
        processed_data = self.get_preprocessed_data(self.train_data)
        grid_search.fit(processed_data[0], processed_data[1])

        self.best_para = grid_search.best_params_

        del grid_search, processed_data

    def training_model(self, data : pd.DataFrame = None):
        if self.parameters_dict != None:
            model = self.classifier.set_params(**self.best_para)
        else:
            model = self.classifier

        
        x, y = self.get_preprocessed_data(data)
        # print(f'precessed features: {x.head}')
        fit_model = model.fit(x, y)
        # print(f'training data shape: {data.shape}')
        # print(f'processed data shape: {x.shape}')
        # print(f'# of features in classifier: {fit_model.n_features_in_}')
        # print('-'*100)
        # self.shape = pd.concat((self.shape, y.value_counts()), axis = 1)
        # x.columns = data.columns[0:-1]
        
        del x, y, model, data
        return fit_model

    def cross_validation(self):
        # Initializing column of the dataset
        variable_names = [str(x) for x in np.arange(self.train_data.shape[1] - 1)]
        variable_names.append('Class')
        self.train_data.columns = variable_names
        
        # Spliting dataset
        kf = StratifiedKFold(n_splits = self.kfold, shuffle = True, random_state = 777)
        train_list = []
        test_list = []
        # Get K-Folds index 
        for train, test in kf.split(self.train_data.iloc[:,0:-1], self.train_data.iloc[:, -1]):
            train_list.append(train)
            test_list.append(test)

        # Cross validation
        metrics_list = ['Accuracy', 'AUC', 'F1-score', 'Recall', 'Precision', 'G-mean']
        evl_outcome = np.zeros((self.kfold, len(metrics_list))) #K-folds, metrics = 6
        for fold in range(self.kfold):
            train_id = train_list[fold]
            test_id = test_list[fold]
            training = self.train_data.iloc[train_id, 0:-1]
            testing = self.train_data.iloc[test_id, 0:-1]

        # Standardization or Normalization
            if self.scaling != None:
                if self.scaling == 'MinMax':
                    scaler = MinMaxScaler().fit(training)
                elif self.scaling == 'Std':
                    scaler = StandardScaler().fit(training)

                training = pd.DataFrame(scaler.transform(training), columns = variable_names[:-1])
                testing =  pd.DataFrame(scaler.transform(testing), columns = variable_names[:-1])
                training['Class'] = self.train_data.iloc[train_id, -1].reset_index(drop = True)
                testing['Class'] = self.train_data.iloc[test_id, -1].reset_index(drop = True)
                del scaler
            else:
                training['Class'] = self.train_data.iloc[train_id, -1]
                testing['Class'] = self.train_data.iloc[test_id, -1]
            
            model = self.training_model(data = training)
            
            
            # if 'best_score' in self.resampler.__dir__():
            #     print(f'best score : {self.resampler.best_score}')
            #     # self.resampler.set_params(**{'best_score' : 0})
            # if 'saturate_count' in self.resampler.__dir__():
            #     print(f'saturate_count : {self.resampler.saturate_count}')
            #     # self.resampler.set_params(**{'saturate_count' : 0})
            # if 'best_balanced_data' in self.resampler.__dir__():
            #     self.resampler.set_params(**{'best_balanced_data' : None})

            testing.columns = training.columns
            pred = model.predict(testing.iloc[:, 0:-1])
            true = testing.iloc[:, -1]
            del model
                
            evl_outcome[fold, :] = list(self.get_metrics_value(true, pred).values())
            del true, pred, train_id, test_id, training, testing 
        print('-'*100)
        return evl_outcome

    def get_metrics_value(self, true, pred):

        metrics_dict = {
            'Accuracy' : accuracy_score(true, pred),
            'AUC' : roc_auc_score(true, pred), 
            'F1-score' : f1_score(true, pred, zero_division = 0),
            'Recall' : recall_score(true, pred, zero_division = 0),
            'Precision' : precision_score(true, pred, zero_division = 0), 
            'G-mean' : geometric_mean_score(true, pred,),
            }

        return metrics_dict
        
    @timer
    def evaluation(self):
        '''Return the evaluation outcome of the given dataset based on the parameter settings
        of the `Train_model` module.
        '''

        metrics_name =   ['Accuracy', 'AUC', 'F1-score', 'Recall', 'Precision', 'G-mean'] 
        evl_table = np.zeros((1, len(metrics_name)))
        if self.parameters_dict != None:
            self.tune_para()
            
        table = self.cross_validation()
        evl_table[0, :] = table.mean(axis = 0)
        self.evaluation_outcome = pd.DataFrame(evl_table, columns = metrics_name)

        return pd.DataFrame(evl_table, columns = metrics_name)
    
    def set_params(self, **parameters : dict):
        for para_name in parameters.keys():
            self.__dict__[para_name] = parameters[para_name]

        # {'train_data': None, 'classifier': None, 'parameters_list': None, 'resampler': None}
