import numpy as np
from matplotlib import pyplot as plt
class Linsep_Binary_Normal_Dataset(object):
    def __init__(self, num:int=128, center:tuple=( 0 , 0 ), deviation:tuple=( 3.9 / np.sqrt(2) , 3.9 / np.sqrt(2) ), stddev:float=1., random_state=39):
        self.__numhalf = num
        assert len(center) == len(deviation) > 0, "Dimension of Center & Deviation must the same Integer!!"
        self.__numdim = len(center)
        self.__center = np.array(center)
        self.__deviation = np.array(deviation)
        self.__stddev = stddev
        rs = np.random.RandomState(random_state)
        self.__X = np.concatenate([
            rs.randn( self.__numdim , self.__numhalf ) * self.__stddev + (self.__center + self.__deviation).reshape( self.__numdim , 1 ),
            rs.randn( self.__numdim  ,self.__numhalf ) * self.__stddev + (self.__center - self.__deviation).reshape( self.__numdim , 1 ),
            ], axis=0)
        self.__y = np.array([+1] * self.__numhalf + [-1] * self.__numhalf).reshape( 1 , self.__numhalf * 2 )
        del rs
    @property
    def center(self):       return self.__center.copy()
    @property
    def deviation(self):    return self.__deviation.copy()
    @property
    def stddev(self):       return self.__stddev
    @property
    def X(self):            return self.__X.copy()
    @property
    def y(self):            return self.__y.copy()
    # Plotting Data in Specified two Dimensions
    def dataplot(self, axis=( 0 , 1 ), labels=[ "pos" , "neg" ]):
        for eachaxis in axis:
            assert 0 <= eachaxis < self.__numdim, ""
        plt.scatter(x=self.__X[ axis[0] , : self.__numhalf ], y=self.__X[ axis[1] , : self.__numhalf ], label=labels[0])
        plt.scatter(x=self.__X[ axis[0] , self.__numhalf : ], y=self.__X[ axis[1] , self.__numhalf : ], label=labels[1])
LBND = Linsep_Binary_Normal_Dataset