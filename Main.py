

from LoadData import *
from TrainModel import *
from Plot import *
from utils.Save_to_file import *
import datetime


# initializations
ld = LoadData()
trainer = TrainModel()
plot = Plot()


# loading data params
'''
Standardizing the inputs can make training faster and reduce the chances of getting stuck in local optima. 
Also, weight decay and Bayesian estimation can be done more conveniently with standardized inputs.
good resources on the topic :
http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
https://ahmedhanibrahim.wordpress.com/2014/10/10/data-normalization-and-standardization-for-neural-networks-output-classification/
https://stackoverflow.com/questions/39930962/do-you-need-to-standardize-inputs-if-you-are-using-batch-normalization
'''
input_standardization = [('minmax',-1,1), ('minmax',0,1), False, 'zscore']
dataset_path = './data/'
dataset_filename = 'breast-cancer-wisconsin-data'
dataset_extension = '.csv'

# training network params
nb_validation_split = 1  # if > 1, then it will use kfold validation
epochs = 1
shuffle = False  # Shuffle don't have any effect if nb_validation_split = 1. It will always pick a 33% test set randomly
nb_neurone_in_each_hidden_layers = [50]
batch_size = [5]
optimizer = 'Adam'
activation_function = 'hard_sigmoid'  # https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0
# first use Relu. If the val_acc is constant (Dying Relu Problem) (Example in the breast-cancer-wisconsin-data.csv with a shallow network and 100 epochs) then use Sigmoid,
weight_initialization = 'glorot_uniform'  # https://towardsdatascience.com/deep-learning-best-practices-1-weight-initialization-14e5c0295b94 TODO : Initialise weight according to this link

# https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82
# https://stats.stackexchange.com/questions/188925/feature-standardization-for-convolutional-network-on-sparse-data
# I don't know a scenario when batch normalization should be disabled
batch_normalization = [True]

for current_input_stardardization_method in input_standardization:

    # loaded_table, input, output = ld.load('./data/TechCrunchcontinentalUSA.csv', columns_to_discard=['permalink', 'company', 'category', 'city', 'fundedDate', 'raisedCurrency', 'round'], columns_to_encode=['state'], independent_variable='raisedAmt')
    # loaded_table, input, output = ld.load('./data/pima-indians-diabetes.data.csv')
    # loaded_table, input, output = ld.load('./data/iris.csv', columns_to_hot_encode=['species'], independent_variable='species')
    loaded_table, input, output = ld.load(path=dataset_path+dataset_filename+dataset_extension, columns_to_discard=['id'], columns_to_hot_encode=['diagnosis'], independent_variable='diagnosis', input_standardization=current_input_stardardization_method)


    # model, history = trainer.train(input, output, epochs=2, nb_validation_split=3, shuffle=True)
    model, results = trainer.train(input, output, epochs=epochs, nb_validation_split=nb_validation_split, shuffle=shuffle,
                                   nb_neurone_in_each_hidden_layers=nb_neurone_in_each_hidden_layers, batch_size=batch_size,
                                   activation_function=activation_function, weight_initialization=weight_initialization)
    # print(history)
    # print(history.history.keys())

    # train duration ???????????

    SaveToFile.save_to_excel('./output/' + dataset_filename + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '.xls', 'result_sheet', results)

    # if nb_validation_split == 1:
    #     plot.plot([history.history['acc'], history.history['val_acc']], title='model accuracy', x_label='epoch', y_label='accuracy')
    #     plot.show_plot()
    #
    #     plot.plot([history.history['loss'], history.history['val_loss']], title='model loss', x_label='epoch', y_label='loss')
    #     plot.show_plot()
    # else:
    #     plot.plot([history['accuracy']], title='model accuracy', x_label='fold n°', y_label='accuracy')
    #     plot.show_plot()


