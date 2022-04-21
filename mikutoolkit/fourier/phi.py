import numpy as np
from matplotlib import pyplot as plt
class Series(object):
    def __init__(self, arr: np.ndarray, numiter:int=6) -> None:
        self.__arr = np.array(arr).flatten()
        self.__numiter = numiter
        # Parsing Parameters
        self.__term = len(self.arr)
        self.__tarr = np.arange(self.__term).astype(int)
        self.__tricoef = np.array([ (k + 1) * 2 * np.pi / self.__term for k in range(self.__numiter) ])
        self.__params = np.zeros(( 2 , self.__numiter ))
        for k in range(self.__numiter):
            self.__params[ 0 , k ] += sum(self.__arr[l] * np.cos((l + .5) * self.__tricoef[k]) for l in self.__tarr)
            self.__params[ 1 , k ] += sum(self.__arr[l] * np.sin((l + .5) * self.__tricoef[k]) for l in self.__tarr)
    @property
    def arr(self) -> np.ndarray:
        return self.__arr.copy()
    @property
    def numiter(self) -> int:
        return self.__numiter
    @property
    def parameters(self) -> np.ndarray:
        return self.__params.copy()
    # Appended Iterations
    def append(self, numiter: int):
        self.__tricoef = np.concatenate([ self.__tricoef, [ ((k + 1 + self.__numiter) * 2 * np.pi / self.__term) for k in range(numiter) ] ])
        self.__params = np.concatenate([ self.__params, np.zeros(( 2 , numiter )) ], axis=1)
        for k in range(numiter):
            self.__params[ 0 , k + self.__numiter ] += sum(self.__arr[l] * np.cos((l + .5) * self.__tricoef[k + self.__numiter]) for l in self.__tarr)
            self.__params[ 1 , k + self.__numiter ] += sum(self.__arr[l] * np.sin((l + .5) * self.__tricoef[k + self.__numiter]) for l in self.__tarr)
        self.__numiter += numiter
        return self
    # Value Estimation
    def fval(self, x):
        return sum(
            self.__params[ 0 , k ] * np.cos(x * self.__tricoef[k]) + self.__params[ 1 , k ] * np.sin(x * self.__tricoef[k])
            for k in range(self.__numiter)
            )
    # Plotting
    def plotarray(self, *args, **kwargs):
        return plt.plot(self.__tarr, self.fval(self.__tarr), *args, **kwargs)