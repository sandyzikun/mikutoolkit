import numpy as np
from matplotlib import pyplot as plt
class Series(object):
    def __init__(ミク, arr: np.ndarray, numiter:int=6) -> None:
        ミク.__arr = np.array(arr).flatten()
        ミク.__numiter = numiter
        # Parsing Parameters
        ミク.__term = len(ミク.arr)
        ミク.__tarr = np.arange(ミク.__term).astype(int)
        ミク.__tricoef = np.array([ (k + 1) * 2 * np.pi / ミク.__term for k in range(ミク.__numiter) ])
        ミク.__params = np.zeros(( ミク.__numiter , 2 ))
        for k in range(ミク.__numiter):
            ミク.__params[ k , 0 ] += sum(ミク.__arr[l] * np.cos((l + .5) * ミク.__tricoef[k]) for l in ミク.__tarr)
            ミク.__params[ k , 1 ] += sum(ミク.__arr[l] * np.sin((l + .5) * ミク.__tricoef[k]) for l in ミク.__tarr)
    @property
    def arr(ミク) -> np.ndarray:
        return ミク.__arr.copy()
    @property
    def numiter(ミク) -> int:
        return ミク.__numiter
    @property
    def parameters(ミク) -> np.ndarray:
        return ミク.__params.copy()
    # Appended Iterations
    def append(ミク, numiter: int):
        ミク.__tricoef = np.concatenate([ ミク.__tricoef, [ ((k + 1 + ミク.__numiter) * 2 * np.pi / ミク.__term) for k in range(numiter) ] ])
        ミク.__params = np.concatenate([ ミク.__params, np.zeros(( numiter , 2 )) ], axis=0)
        for k in range(numiter):
            ミク.__params[ k + ミク.__numiter , 0 ] += sum(ミク.__arr[l] * np.cos((l + .5) * ミク.__tricoef[k + ミク.__numiter]) for l in ミク.__tarr)
            ミク.__params[ k + ミク.__numiter , 1 ] += sum(ミク.__arr[l] * np.sin((l + .5) * ミク.__tricoef[k + ミク.__numiter]) for l in ミク.__tarr)
        ミク.__numiter += numiter
        return ミク
    # Value Estimation
    def fval(ミク, x):
        return sum(
            ミク.__params[ k , 0 ] * np.cos(x * ミク.__tricoef[k]) + ミク.__params[ k , 1 ] * np.sin(x * ミク.__tricoef[k])
            for k in range(ミク.__numiter)
            )
    # Plotting
    def plotarray(ミク, *args, **kwargs):
        return plt.plot(ミク.__tarr, ミク.fval(ミク.__tarr), *args, **kwargs)