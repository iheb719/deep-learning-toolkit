from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.constraints import maxnorm

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from machine_learning.ml_utils import MlUtils

import sklearn
import random


# todo : Add genetic algorith to automatically select network params
class TrainModel:

    model = Sequential()

    def train(self, input, output, epochs=[10], nb_validation_split=10, shuffle=False,
              nb_neurone_in_each_hidden_layers=[10], batch_size=[5],
              activation_function=['relu'],  # activation function for all layer except the last one
              weight_initialization=['glorot_uniform'],
              optimizer=['Adam'], batch_normalization=['BeforeActivation'],
              dropout_rate=[0.2], weight_constraint=[3]):

        if shuffle:
            random_state = random.randint(1, 521364)
            input = self.shuffle_rows(input, random_state)
            output = self.shuffle_rows(output, random_state)

        #  sigmoid vs softmax : https://stats.stackexchange.com/a/254071
        # if the independent varible is composed of 1 column
        if len(output.shape) == 1 or output.shape[1] == 1:
            loss_fct = 'binary_crossentropy'
            output_layer_activation = 'sigmoid'
        else:
            loss_fct = 'categorical_crossentropy'
            output_layer_activation = 'softmax'

        # if nb_validation_split == 1:  # Automatic Verification Dataset (not kflod)
        #     output_dim = 1 if len(output.shape) == 1 else output.shape[1]
        #     model = self.get_model(input_dim=input.shape[1], output_dim=output_dim, loss_fct=loss_fct,
        #                            output_layer_activation=output_layer_activation,
        #                            nb_neurone_in_each_hidden_layers=nb_neurone_in_each_hidden_layers,
        #                            activation_function=activation_function,
        #                            weight_initialization=weight_initialization)
        #     history = model.fit(input, output, validation_split=0.33, epochs=epochs, batch_size=batch_size)
        # else:  # kfold validation

        # when the input data is small or if you have sufficient compute resources use kfold validation
        output_dim = 1 if len(output.shape) == 1 else output.shape[1]
        param_grid = dict(input_dim=[input.shape[1]], output_dim=[output_dim], loss_fct=[loss_fct],
                          output_layer_activation=[output_layer_activation], epochs=epochs,
                          nb_neurone_in_each_hidden_layers=nb_neurone_in_each_hidden_layers,
                          activation_function=activation_function, weight_initialization=weight_initialization,
                          batch_size=batch_size, optimizer=optimizer, batch_normalization=batch_normalization,
                          dropout_rate=dropout_rate, weight_constraint=weight_constraint)
        model = KerasClassifier(build_fn=self.get_model, verbose=10)

        cv = MlUtils.get_cross_validation(nb_validation_split, shuffle, output.shape[1])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        grid_result = grid.fit(input, output)

        MlUtils.print_gridsearch_results(grid_result)

        return model, grid_result

        # val_list = []
        #
        # # print((current_key, current_rst) for current_key, current_rst in grid_result.cv_results_.items() if current_key.endswith('test_score'))
        #
        # # get all values from the splitXXX_test_score parameter
        # for idx, (current_key, current_rst) in enumerate([(current_key, current_rst) for current_key, current_rst in grid_result.cv_results_.items() if current_key.endswith('test_score') and current_key.startswith('split')]):
        #     val_list.extend(current_rst)
        # print(val_list)
        # history = dict(accuracy=val_list)
        #
        # return model, history

    def get_model(self, input_dim=8, output_dim=1, loss_fct='binary_crossentropy', output_layer_activation='sigmoid',
                  nb_neurone_in_each_hidden_layers=[10], activation_function='relu', weight_initialization='glorot_uniform',
                  optimizer='Adam', batch_normalization='BeforeActivation',
                  dropout_rate=[0.2], weight_constraint=[3]):
        model = Sequential()
        # model.add(Dense(8, input_shape=(input.shape[1],), kernel_initializer='uniform', activation='relu'))
        # #model.add(Dense(4, activation='relu'))
        # #model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        # #model.add(Dense(10, activation='relu'))
        # #model.add(Dense(4, activation='relu'))
        # model.add(Dense(output.shape[1], kernel_initializer='uniform', activation=output_layer_activation))
        # model.compile(optimizer=Adam(lr=0.001), loss=loss_fct, metrics=['accuracy'])
        for idx, nb_neurone_in_current_hidden_layer in enumerate(nb_neurone_in_each_hidden_layers):
            if idx == 0:
                # first layer + first hidden layer
                # https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/
                # --> To make it Batch normalization enabled, we have to tell the Dense layer not using bias since it is not needed, it can save some calculation
                model.add(Dense(nb_neurone_in_current_hidden_layer, input_dim=input_dim, kernel_initializer=weight_initialization,
                                use_bias=True if not batch_normalization else False,
                                kernel_constraint=weight_constraint))
                if batch_normalization in ['BeforeActivation', 'AroundActivation'] :
                    model.add(BatchNormalization())
                model.add(Activation(activation=activation_function))
                if batch_normalization in ['AfterActivation', 'AroundActivation'] :
                    model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            else:
                # other hidden layers
                model.add(Dense(nb_neurone_in_current_hidden_layer, kernel_initializer=weight_initialization,
                                use_bias=True if not batch_normalization else False,
                                kernel_constraint=weight_constraint))
                if batch_normalization in ['BeforeActivation', 'AroundActivation']:
                    model.add(BatchNormalization())
                model.add(Activation(activation=activation_function))
                if batch_normalization in ['AfterActivation', 'AroundActivation'] :
                    model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))

        # output layer
        model.add(Dense(output_dim, use_bias=True if not batch_normalization else False))
        if batch_normalization in ['BeforeActivation', 'AroundActivation']:
            model.add(BatchNormalization())
        model.add(Activation(activation=activation_function))
        if batch_normalization in ['AfterActivation', 'AroundActivation']:
            model.add(BatchNormalization())
        # I didn't see example of using dropout in the last layer

        # Compile model
        model.compile(loss=loss_fct, optimizer=optimizer, metrics=['accuracy'])
        return model

    def train33(self, input, output, epochs=10):
        model = Sequential()
        model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        history = model.fit(input, output, validation_split=0.33, epochs=100, batch_size=10, verbose=0)

        return model, history


    def get_predicted_values(self, model, input):
        return model.predict(input)

    def evaluate_model(self, model):
        print('')


    def evaluate_single_result(self, model, input, output):
        if(model.predict(input) == output):
            return True

        return False

    def shuffle_rows(self, datasource, random_state):

        datasource_shuffled = sklearn.utils.shuffle(datasource, random_state=random_state)

        return datasource_shuffled