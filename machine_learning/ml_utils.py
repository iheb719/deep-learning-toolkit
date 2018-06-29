
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit


class MlUtils:

    @staticmethod
    def get_cross_validation(nb_validation_split=1, shuffle=False, output_shape=1):
        if nb_validation_split == 1:  # no cross validation
            cv = ShuffleSplit(test_size=0.33, n_splits=1)
        else:  # cross validation
            if shuffle:
                cv = StratifiedShuffleSplit(n_splits=nb_validation_split)
            else:
                if output_shape == 1:
                    cv = StratifiedKFold(n_splits=nb_validation_split, shuffle=False)
                else:  # StratifiedKFold cant be used in multiclass classification
                    cv = KFold(n_splits=nb_validation_split, shuffle=False)

        return cv

    @staticmethod
    def print_gridsearch_results(grid_result):
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))