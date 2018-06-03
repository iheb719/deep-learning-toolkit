
import pandas as pd
from sklearn import preprocessing
from scipy.stats import zscore

class Standardization:

    @staticmethod
    def algorithm_selection(datasource, algorithm):

        if type(algorithm) == str:
            if algorithm.startswith('zscore'):
                return Standardization.minmax(datasource)

        if type(algorithm) == tuple:
            if algorithm[0] == 'minmax':
                return Standardization.minmax(datasource, algorithm[1], algorithm[2])

    @staticmethod
    def minmax(datasource, min=0, max=1):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(min, max))
        np_scaled = min_max_scaler.fit_transform(datasource)
        loaded_table = pd.DataFrame(np_scaled, columns=datasource.columns)
        print(f' normalized table minmax with min {min} and max {max} : ', loaded_table)
        return loaded_table


    @staticmethod
    def zscore(datasource):
        standardized_data = datasource.apply(zscore)
        print(' normalized table zscore : ', standardized_data)
        return standardized_data