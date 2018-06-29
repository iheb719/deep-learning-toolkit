from load_data import *
from machine_learning.svm import *
from plot import *
from common_utils.save_to_file import *
import datetime


# initializations
ld = LoadData()
trainer = svm()
plot = Plot()


# loading data params
input_standardization = [('minmax', 0, 1)]
dataset_path = './data/'
dataset_filename = 'iris'
dataset_extension = '.csv'

# training network params
nb_validation_split = [10, 5]  # if > 1, then it will use kfold validation
shuffle_dataset = [False]  # Shuffle don't have any effect if nb_validation_split = 1. It will always pick a 33% test set randomly

# Tips For Using Dropout : https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
c = [1]
gamma = [1]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']

# variable utils
output_file_path = None
first_iteration = True

for current_input_stardardization_method in input_standardization:

    # loaded_table, input, output = ld.load('./data/TechCrunchcontinentalUSA.csv', columns_to_discard=['permalink', 'company', 'category', 'city', 'fundedDate', 'raisedCurrency', 'round'], columns_to_encode=['state'], independent_variable='raisedAmt')
    # loaded_table, input, output = ld.load('./data/pima-indians-diabetes.data.csv')
    loaded_table, input, output = ld.load(path=dataset_path+dataset_filename+dataset_extension, columns_to_hot_encode=['species'], independent_variable='species')
    # loaded_table, input, output = ld.load(path=dataset_path+dataset_filename+dataset_extension, columns_to_discard=['id'], columns_to_hot_encode=['diagnosis'], independent_variable='diagnosis', input_standardization=current_input_stardardization_method)

    for current_nb_validation_split in nb_validation_split:
        for current_shuffle in shuffle_dataset:
            model, results = trainer.train(input, output, nb_validation_split=current_nb_validation_split,
                                           shuffle_dataset=current_shuffle, gamma=gamma,
                                           c=c, kernel=kernel,
                                           #degree=degree
                                           )
            # print(history)
            # print(history.history.keys())

            # TODO : train duration

            # format results to save to file
            formatted_results = SaveToFile.format_gridsearchcv_result(results,
                       [('Input stardardization method', current_input_stardardization_method),
                        ('Nb validation split', current_nb_validation_split),
                        ('Shuffle', current_shuffle)
                       ]
                                                                       )

            output_file_path = output_file_path if output_file_path != None else './output/' + dataset_filename + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '.xls'
            SaveToFile.save_to_excel(output_file_path, 'result_sheet', formatted_results, not first_iteration, not first_iteration)
            first_iteration = False

    # if nb_validation_split == 1:
    #     plot.plot([history.history['acc'], history.history['val_acc']], title='model accuracy', x_label='epoch', y_label='accuracy')
    #     plot.show_plot()
    #
    #     plot.plot([history.history['loss'], history.history['val_loss']], title='model loss', x_label='epoch', y_label='loss')
    #     plot.show_plot()
    # else:
    #     plot.plot([history['accuracy']], title='model accuracy', x_label='fold n°', y_label='accuracy')
    #     plot.show_plot()


