import numpy as np
from matplotlib import pyplot as plt
class Linsep_Binary_Normal_Dataset(object):
    def __init__(ミク, num:int=128, center:tuple=( 0 , 0 ), deviation:tuple=( 3.9 / np.sqrt(2) , 3.9 / np.sqrt(2) ), stddev:float=1., random_state=39):
        ミク.__numhalf = num
        assert len(center) == len(deviation) > 0, "Dimension of Center & Deviation must the same Integer!!"
        ミク.__numdim = len(center)
        ミク.__center = np.array(center)
        ミク.__deviation = np.array(deviation)
        ミク.__stddev = stddev
        rs = np.random.RandomState(random_state)
        x0 = rs.randn( ミク.__numhalf * 2 * ミク.__numdim ) * ミク.__stddev
        ミク.__X = np.zeros(( ミク.__numdim , ミク.__numhalf * 2 ))
        for k in range(ミク.__numhalf):
            ミク.__X[ : , k ] += x0[ (k * ミク.__numdim) : ((k+1) * ミク.__numdim) ] + (ミク.__center + ミク.__deviation)
        for k in range(ミク.__numhalf, ミク.__numhalf * 2):
            ミク.__X[ : , k ] += x0[ (k * ミク.__numdim) : ((k+1) * ミク.__numdim) ] + (ミク.__center - ミク.__deviation)
        ミク.__y = np.repeat([ +1 , -1 ], ミク.__numhalf).reshape( ミク.__numhalf * 2 , 1 )
        del rs
    @property
    def center(ミク):       return ミク.__center.copy()
    @property
    def deviation(ミク):    return ミク.__deviation.copy()
    @property
    def stddev(ミク):       return ミク.__stddev
    @property
    def X(ミク):            return ミク.__X.copy()
    @property
    def y(ミク):            return ミク.__y.copy()
    # Plotting Data in Specified two Dimensions
    def dataplot(ミク, axis=( 0 , 1 ), labels=[ "pos" , "neg" ]):
        for eachaxis in axis:
            assert 0 <= eachaxis < ミク.__numdim, ""
        plt.scatter(x=ミク.__X[ axis[0] , : ミク.__numhalf ], y=ミク.__X[ axis[1] , : ミク.__numhalf ], label=labels[0])
        plt.scatter(x=ミク.__X[ axis[0] , ミク.__numhalf : ], y=ミク.__X[ axis[1] , ミク.__numhalf : ], label=labels[1])
LBND = Linsep_Binary_Normal_Dataset