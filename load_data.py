
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from common_utils.stardization import Standardization

from typing import List


class LoadData:
    'Common base class for all employees'
    empCount = 0

    'def __init__(self):'
    '    LoadData.empCount += 1'

    def displayCount(self):
        print("Total Employee %d")

    @staticmethod
    def hot_encode(data):
        encoder = LabelBinarizer()
        transformed_label = encoder.fit_transform(data)
        return transformed_label

    @staticmethod
    def hot_encode_v2(data):  # just another way to do it
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(data)
        encoded_Y = encoder.transform(data)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        return dummy_y

    def load(self, path:str, columns_to_discard:List[str]='', columns_to_hot_encode:List[str]='', independent_variable:str='', input_standardization:bool=False):
        loaded_table = self.load_file_by_type(path)
        # loaded_table = loaded_table[(loaded_table['State'] == 'California')]
        # encoded_columns = LoadData.encode(loaded_table.iloc[:, [3]])
        # take only 2 out of 3 states to avoid the dummy variable trap
        #loaded_table.insert(loc=3, column='State1', value=encoded_columns[:, [0]])
        #loaded_table.insert(loc=4, column='State2', value=encoded_columns[:, [1]])

        if columns_to_discard != '':
            loaded_table = LoadData.drop_columns(loaded_table, columns_to_discard)

        loaded_table = LoadData.drop_lines_containing_null_values(loaded_table)

        if columns_to_hot_encode != '':
            loaded_table = LoadData.hot_encode_columns(loaded_table, columns_to_hot_encode, independent_variable)

        if input_standardization:
            loaded_table = Standardization.algorithm_selection(loaded_table, input_standardization)

        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        print(loaded_table)

        if independent_variable != '':
            # https://medium.freecodecamp.org/python-list-comprehensions-vs-generator-expressions-cef70ccb49db
            dependent_variable_columns = [col for col in loaded_table if not col.startswith(independent_variable)]
            input = loaded_table[dependent_variable_columns]

            independent_variable_columns = [col for col in loaded_table if col.startswith(independent_variable)]
            output = loaded_table[independent_variable_columns]
        else:
            input = loaded_table.iloc[:, :-1]  # don't take the last column
            output = loaded_table.iloc[:, -1]  # take only the last column

        return loaded_table, input, output

    @staticmethod
    def load_file_by_type(path):
        """ load a file and returns a panda dataframe """
        if path is None:
            raise ValueError('input file is empty')

        if path.endswith('.csv'):
            return pd.read_csv(filepath_or_buffer=path, skiprows=0)

        if path.endswith('.arff'):  # weka files
            raw_data = loadarff(path)
            return pd.DataFrame(raw_data[0])

        raise ValueError('input file type not supported')

    @staticmethod
    def drop_columns(loaded_table, columns_to_drop):

        for current_column in columns_to_drop:
            loaded_table = loaded_table.drop([current_column], axis=1)
            print('Dropped Column : ', current_column)

        print('table after drop ', loaded_table)
        return loaded_table

    @staticmethod
    def drop_lines_containing_null_values(loaded_table):

        for name, values in loaded_table.iteritems():
            loaded_table = loaded_table[(loaded_table[name].notnull())]
            #loaded_table = loaded_table[(loaded_table['State'] == 'California')]
            print('Dropped Line')

        print('table after dropping line', loaded_table)
        return loaded_table

    @staticmethod
    def hot_encode_columns(loaded_table, columns_to_encode, independent_variable):

        if isinstance(columns_to_encode, str):
            columns_to_encode = [columns_to_encode]

        for current_column in columns_to_encode:
            # c = loaded_table[[current_column]]
            current_column_encoded = LoadData.hot_encode(loaded_table[[current_column]])
            current_column_encoded = np.transpose(current_column_encoded)

            print(current_column_encoded[0])

            dummy_variable_trap = 1  # don't include the last column to avoid the dummy variable trap
            if current_column == independent_variable:
                dummy_variable_trap = 0  # if its an independent variable, all columns must be included

            for i in range(0, current_column_encoded.shape[0]-dummy_variable_trap):
                loaded_table[current_column + str(i)] = current_column_encoded[i]
                # loaded_table.insert(loc=3, column='State'+str(i), value=aaaa)

            loaded_table = loaded_table.drop([current_column], axis=1)
            print('Drop Column after encoding it : ', current_column)

        print('table after encoding', loaded_table)
        return loaded_table