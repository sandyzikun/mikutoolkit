import numpy as np
class Linear_Single_Preceptron(object):
    def __init__(self, numunits: int, lr:float=.1):
        self.__num_units = numunits
        self.__weights = np.ones(( 1 , self.__num_units ))
        self.__bias = 0.
        self.__learning_rate = lr
    def predict(self, x: np.ndarray) -> int:
        """
        x: Row-vector shaped in ( 1 , num_units )
        """
        return np.sign(self.__weights @ x.T + self.__bias)
    def fit_sample(self, x: np.ndarray, y: int):
        """
        y: Integer in { -1 , +1 }
        """
        if y * self.predict(x) <= 0:
            self.__weights += self.__learning_rate * y * x
            self.__bias    += self.__learning_rate * y
        return self
    def fit(self, x: np.ndarray, y: np.ndarray):
        for k in range(x.shape[0]):
            self = self.fit_sample(x[ k : (k + 1) , : ], y[ k , 0 ])
        return self
    def acc(self, x: np.ndarray, y: np.ndarray):
        res = 0
        for k in range(x.shape[0]):
            if self.predict(x[ k : (k + 1) , : ]) == y[ k , 0 ]:
                res += 1
        return res / x.shape[0]
LSP = Linear_Single_Preceptron
class Fool_Connection(object):
    def __init__(self, numunits: int, numinput: int, lr:float=.1):
        self.__num_input = numinput
        self.__num_units = numunits
        self.__learning_rate = lr
        self.__weights = np.random.randn()