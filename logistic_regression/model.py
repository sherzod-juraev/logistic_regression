from numpy import ndarray, exp as nm_exp, clip as nm_clip, dot as nm_dot, add as nm_add, where as nm_where, subtract as nm_subtract, log as nm_log
from numpy.random import uniform as nm_uniform


class LogisticRegressionGD:
    """ Logistic Regression using gradient descent

    Parameters
    ------------

        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
    """

    def __init__(self, eta: float = .01, n_iter: int = 100):

        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.b = nm_uniform(-.01, .01)
        self.eps = 1e-6

    def initialize_weights(self, length: int, /):

        self.w_ = nm_uniform(-.01, .01, size=length)

    def fit(self, X: ndarray, y: ndarray, /) -> bool:
        """Fit training data

        Parameters
        -----------

        X:  {array-like},
            shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y : array-like,
            shape = [n_examples]
            Target values.

        Return
        -------

        result: bool,
            Returns true if the model can fit the data without reaching
             the n_iter limit, false if it stops after reaching n_iter
        """

        self.initialize_weights(X.shape[1])
        n_sample = X.shape[0]
        J_last, J_old = None, None
        for i in range(self.n_iter):
            z = self.net_input(X)
            sigmoid = self.sigmoid(z)
            errors = nm_subtract(y, sigmoid)
            self.w_ += self.eta * nm_dot(X.T, errors)
            self.b += self.eta * errors.sum()
            J_last = (1 / n_sample) * (nm_dot(y, nm_log(sigmoid)) + nm_dot(nm_subtract(1, y), nm_log(nm_subtract(1, sigmoid)))).sum()
            if i != 0:
                if abs(J_last - J_old) <= self.eps:
                    return True
            J_old = J_last
        return False


    def net_input(self, X: ndarray, /) -> ndarray:
        """Calculate net input"""

        net_input = nm_add(nm_dot(X, self.w_), self.b)
        return net_input

    def sigmoid(self, z: ndarray, /) -> ndarray:
        """Compute logistic sigmoid activation"""

        sigmoid = 1 / nm_add(1, nm_exp(-nm_clip(z, -250, 250)))
        return sigmoid

    def predict(self, X: ndarray, /) -> int:
        """Return class label after unit step"""

        sigmoid = self.sigmoid(self.net_input(X))
        result = nm_where(sigmoid >= .5, 1, 0)
        return result