import matplotlib.pyplot as plt


class Plot:

    def plot_x_y(self, x, y):
        plt.plot(x, y)
        plt.legend(loc='upper left')

    def show_plot(self, legend='', legend_position='upper left'):
        if legend != '':
            plt.legend(legend, loc=legend_position)
        plt.show()

    def plot(self, ds_list, title='', x_label='', y_label='', legend=['train', 'test'], legend_position='upper left'):
        for current_ds in ds_list:
            plt.plot(current_ds)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend(legend, loc=legend_position)

    def plot_accuracy(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def plot_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
