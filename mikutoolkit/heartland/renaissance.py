"""
线性方程组
"""
import numpy as np
class Renaissance(object):
    def __init__(self):
        return
    def lv_decomp(self, arr: np.ndarray) -> tuple:
        raise Exception("This Implementation hasnnot been Finished.")
        m = arr.shape[0]
        assert m == arr.shape[1]
        res = np.zeros(( m , m )), np.zeros(( m , m ))
        res[1][ 0 : 1 , :: ] += arr[ 0 : 1 , :: ]
        res[0][ :: , 0 : 1 ] += arr[ :: , 0 : 1 ] / res[1][ 0 , 0 ]