import numpy as np
class Linear_Single_Preceptron(object):
    def __init__(ミク, numunits: int, lr:float=.1):
        ミク.__num_units = numunits
        ミク.__weights = np.ones(( 1 , ミク.__num_units ))
        ミク.__bias = 0.
        ミク.__learning_rate = lr
    def predict(ミク, x: np.ndarray) -> int:
        """
        x: Row-vector shaped in ( 1 , num_units )
        """
        return np.sign(ミク.__weights @ x.T + ミク.__bias)
    def fit_sample(ミク, x: np.ndarray, y: int):
        """
        y: Integer in { -1 , +1 }
        """
        if y * ミク.predict(x) <= 0:
            ミク.__weights += ミク.__learning_rate * y * x
            ミク.__bias    += ミク.__learning_rate * y
        return ミク
    def fit(ミク, x: np.ndarray, y: np.ndarray):
        for k in range(x.shape[0]):
            ミク = ミク.fit_sample(x[ k : (k + 1) , : ], y[ k , 0 ])
        return ミク
    def acc(ミク, x: np.ndarray, y: np.ndarray):
        res = 0
        for k in range(x.shape[0]):
            if ミク.predict(x[ k : (k + 1) , : ]) == y[ k , 0 ]:
                res += 1
        return res / x.shape[0]
LSP = Linear_Single_Preceptron
class Fool_Connection(object):
    """
    * Activations
      * activation: tanh(x) = (exp(+x) - exp(-x)) / (exp(+x) + exp(-x))
      * derivative: (d tanh) / (dx) = 1 - (tanh(x))^2
    * Tensor Graph: `[X] -> (@W) -> (+b) -> (activate) -> y`
    * Loss: (y_pred - y_true)^2
    """
    def __init__(ミク, numunits: int, numinput: int, lr:float=.1, W0=None, b0=None, random_state=39):
        if W0 is not None:
            assert W0.shape == ( numinput , numunits )
        if b0 is not None:
            assert b0.shape == ( 1 , numunits )
        ミク.__num_input = numinput
        ミク.__num_units = numunits
        ミク.__learning_rate = lr
        rs = np.random.RandomState(random_state)
        ミク.__weights = np.zeros(( numinput , numunits ))
        if W0 is not None:
            ミク.__weights += W0
        else:
            w0 = rs.randn(numinput * numunits)
            for k in range(numinput):
                ミク.__weights[ k , : ] += w0[ k * numunits : (k + 1) * numunits ]
        ミク.__bias = b0 or np.zeros(( 1 , numunits ))
    @property
    def weights(ミク):
        return ミク.__weights.copy()
    @property
    def bias(ミク):
        return ミク.__bias.copy()
    # Prediction
    def predict(ミク, X: np.ndarray):
        assert X.shape == ( 1 , ミク.__num_input )
        return np.tanh((X @ ミク.__weights) + ミク.__bias)
    # Loss Function
    def loss(ミク, y_pred, y_true):
        assert y_pred.shape == y_true.shape == ( 1 , ミク.__num_units )
        return ((y_pred - y_true).flatten() ** 2).reshape( 1 , ミク.__num_units ) / 2
    # Training Process
    def fit(ミク, X: np.ndarray, y: np.ndarray):
        assert X.shape == ( 1 , ミク.__num_input ) \
           and y.shape == ( 1 , ミク.__num_units )
        y_pred = ミク.predict(X)
        y_pred_flat = y_pred.flatten()
        loss_partial_Wxb = [
            (y_pred_flat - y.flatten()),
            (1 - y_pred_flat ** 2),
            ]
        loss_partial_Wxb = loss_partial_Wxb[0] * loss_partial_Wxb[1]
        loss_partial_Wxb = loss_partial_Wxb.reshape( 1 , ミク.__num_units )
        # Update Bias Vector
        ミク.__bias -= ミク.__learning_rate * loss_partial_Wxb
        # Update Weights' Matrix
        loss_partial_W = np.zeros(( ミク.__num_input , ミク.__num_units ))
        for k in range(ミク.__num_input):
            for l in range(ミク.__num_units):
                loss_partial_W[ k , l ] += X[ 0 , k ] * loss_partial_Wxb[ 0 , l ]
        ミク.__weights -= ミク.__learning_rate * loss_partial_W
        return ミク
FC = Fool_Connection