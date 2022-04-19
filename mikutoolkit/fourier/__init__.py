import numpy as np
from matplotlib import pyplot as plt
class Series(object):
    def __init__(self, arr: np.ndarray, numiter:int=6) -> None:
        self.__arr = np.array(arr).flatten()
        self.__numiter = numiter
        # Parsing Parameters
        self.__term = len(self.arr)
        self.__params = []
        self.__tricoef = [ (k + 1) * 2 * np.pi / self.__term for k in range(self.__numiter) ]
        for k in range(self.__numiter):
            self.__params.append((
                sum(self.__arr[l] * np.cos((l + .5) * self.__tricoef[k]) for l in range(self.__term)),
                sum(self.__arr[l] * np.sin((l + .5) * self.__tricoef[k]) for l in range(self.__term)),
                ))
    @property
    def arr(self) -> np.ndarray:
        return self.__arr
    @property
    def numiter(self) -> int:
        return self.__numiter
    @property
    def parameters(self) -> list:
        return self.__params
    # Appended Iterations
    def append(self, numiter: int):
        self.__tricoef += [ ((k + 1 + self.__numiter) * 2 * np.pi / self.__term) for k in range(numiter) ]
        for k in range(numiter):
            self.__params.append((
                sum(self.__arr[l] * np.cos((l + .5) * self.__tricoef[k + self.__numiter]) for l in range(self.__term)),
                sum(self.__arr[l] * np.sin((l + .5) * self.__tricoef[k + self.__numiter]) for l in range(self.__term)),
                ))
        self.__numiter += numiter
        return self
    # Value Estimation
    def fval(self, x):
        return sum(
            self.__params[k][0] * np.cos(x * self.__tricoef[k]) + self.__params[k][1] * np.sin(x * self.__tricoef[k])
            for k in range(self.__numiter)
            )
    # Plotting
    def plotarray(self, *args, **kwargs):
        xlist = list(range(self.__term))
        return plt.plot(xlist, [ self.fval(x + .5) for x in xlist ], *args, **kwargs)