"""

https://stats.stackexchange.com/questions/95340/comparing-svm-and-logistic-regression

Angrew NG : https://www.youtube.com/watch?v=hDh7jmEGoY0
n nb of features, m nb of training examples
if n is large (relative to m)
    use logistic regression or svm without a kernel (linear kernel)
if n is small and m is intermediate (n = 1 --> 1 000 / m = 10 --> 10 000)
    use svm with gaussian kernal
if n is small, m is large (n = 1 --> 1 000 / m > 50 000)
    create or add new features then use logistic regression or svm without a kernel (linear kernel)
neural networks likely to work well for more of these settings but may train slowly

Note : Do perform feature scaling before using the gaussian kernel
polynomial kernal nearly in all cases perform worse than gaussian kernal

Linear SVMs and logistic regression generally perform comparably in practice.
You better choose one of them and learn how to tune their parameter than debating what algorithm is better


Pros:
It works really well with clear margin of separation
It is effective in high dimensional spaces.
It is effective in cases where number of dimensions is greater than the number of samples.
It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
Cons:
It doesn’t perform well, when we have large data set because the required training time is higher
It also doesn’t perform very well, when the data set has more noise i.e. target classes are overlapping
SVM doesn’t directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. It is related SVC method of Python scikit-learn library.

"""


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from machine_learning.ml_utils import MlUtils

class svm:

    """
    https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
    How to tune Parameters of SVM :
        For kernel : Look fir the top comments
        gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Higher the value of gamma, will try to exact fit the as per training data set i.e. generalization error and cause over-fitting problem.
        C:
            https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine
            allows some examples to be "ignored" or placed on the wrong side of the margin
            http://scikit-learn.org/stable/modules/svm.html#svm-regression
            Setting C: C is 1 by default and it’s a reasonable default choice. If you have a lot of noisy observations you should decrease it. It corresponds to regularize more the estimation.
        Degree : polynomial degree if the kernel is poly
    """
    def train(self, input, output, nb_validation_split=1, shuffle_dataset=[True], kernel=['rbf'], degree=[3], gamma=['auto'], c=[1.0]):
        classifier = OneVsRestClassifier(SVC(kernel=kernel, degree=degree, gamma=gamma, coef0=0.0,
                 tol=1e-3, C=c, shrinking=True,
                 verbose=1, max_iter=-1))

        x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.33)

        # TODO : degree parameter should only be used when kernet is poly
        param_grid = dict(estimator__C=c, estimator__gamma=gamma, estimator__kernel=kernel, estimator__degree=degree)

        cv = MlUtils.get_cross_validation(nb_validation_split, shuffle_dataset, output.shape[1])

        print(classifier.get_params().keys())

        grid_search = GridSearchCV(classifier, param_grid, cv=cv)
        grid_result = grid_search.fit(x_train, y_train)

        MlUtils.print_gridsearch_results(grid_result)

        return classifier, grid_result
