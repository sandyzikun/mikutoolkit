import numpy as np
from matplotlib import pyplot as plt
class Poly_Lagrange(object):
    def __init__(ミク, arr: np.ndarray):
        ミク.__arr = arr
        ミク.__length = arr.shape[0]
        ミク.__idxlst = np.arange(arr.shape[0]).astype(int)
        ミク.__deltas = np.array([[
            arr[ k , 0 ] - arr[ l , 0 ]
            for l in ミク.__idxlst
            ]
            for k in ミク.__idxlst
            ])
    def __call__(ミク, x: float) -> float:
        res = []
        for k in ミク.__idxlst:
            res.append(1.)
            for l in ミク.__idxlst:
                if k != l:
                    res[-1] *= (x - ミク.__arr[ l , 0 ]) / ミク.__deltas[ k , l ]
            res[-1] *= ミク.__arr[ k , 1 ]
        return np.sum(res)
    def __len__(ミク) -> int:
        return ミク.__length
    def plot(ミク, arr, *args, **kwargs):
        return plt.plot(arr, [ ミク(each) for each in arr ], *args, **kwargs)
class Poly_Newton(object):
    def __init__(ミク, arr: np.ndarray):
        ミク.__arr = arr
        ミク.__length = arr.shape[0]
        ミク.__idxlst = np.arange(arr.shape[0]).astype(int)
        ミク.__avgdev = arr[ : , 1 ].tolist()
        for k in range(1, ミク.__length):
            cur = [
                (ミク.__avgdev[l] - ミク.__avgdev[l - 1]) / (arr[ l , 0 ] - arr[ (l - k) , 0 ])
                for l in range(k, ミク.__length)
                ]
            ミク.__avgdev[ k : ] = cur
    def __call__(ミク, x: float) -> float:
        res = ミク.__avgdev[-1]
        for k in range(ミク.__length - 2, -1, -1):
            res *= (x - ミク.__arr[ k , 0 ])
            res += ミク.__avgdev[k]
        return res
    def __len__(ミク) -> int:
        return ミク.__length
    def plot(ミク, arr, *args, **kwargs):
        return plt.plot(arr, [ ミク(each) for each in arr ], *args, **kwargs)
class Orbitdance(object):
    def __init__(ミク):
        return
    def poly_interpolate(ミク, samples, method: str):
        arr = np.array(samples)
        assert len(arr.shape) == 2 and arr.shape[1] == 2
        return {
            "lagrange": Poly_Lagrange,
            "newton": Poly_Newton,
            }[method.lower()](arr)
__息吹く炎_君の鼓動の中__ = Orbitdance()
poly_interpolate = __息吹く炎_君の鼓動の中__.poly_interpolate