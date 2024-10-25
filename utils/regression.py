import numpy as np
from utils.base import Function, Optim, Algo
from utils.first_order import GradientDescent
from tqdm import tqdm


class LinearRegression (Algo) :
    def __init__(self, optim : Optim, *args, **kwargs) -> None:
        self.optim = optim 
        self.__is_trained : bool = False 
        self.__W  : np.ndarray | None = None
        self.__B  : np.ndarray | None = None

        self.__Y_cap_func = lambda x_aug, w_aug : x_aug @ w_aug
        self.__Error_func = lambda y, y_cap :  y_cap - y
        return
    
    def _init_params (self, size : tuple) -> None :
        if (not self.__is_trained) : 
            self.__W = np.random.normal(size = size[1]) 
            self.__B = np.random.normal(size = 1)

    def _reset (self) -> None : 
        if (self.__is_trained) : 
            self.__W = None
            self.__B = None
            self.__is_trained = False 

    def train (self, X_train : np.ndarray, Y_train : np.ndarray, epochs : int = 1, is_plot : bool = False) -> None : 
        assert (X_train.shape[0] == Y_train.shape[0]), f"Dimension mismatch X : {X_train.shape} and Y : {Y_train.shape}"

        # if model already trained, reset the parameters
        if (self.__is_trained) : 
            self._reset()

        # initialise random parameters 
        self._init_params(X_train.shape)
        
        assert (isinstance(self.__W,np.ndarray)), "Error in initialising W"
        assert (isinstance(self.__B,np.ndarray)), "Error in initialising B"

        # construction of Augmented matrices
        one_coloums = np.ones((X_train.shape[0], 1))
        X_train_aug = np.concatenate([one_coloums, X_train], axis = 1)
        W_aug = np.concatenate([self.__B, self.__W], axis = 0)


        # training loop
        for epoch in range(epochs)  : 
            
            # initialising Loss Function 
            W_aug.resize((len(W_aug),1))
            loss = Function(
                    func = lambda w : 1/(2*Y_train.shape[0])  * np.linalg.norm (self.__Error_func(Y_train, self.__Y_cap_func(X_train_aug, w))), 
                    name = "mean square error"
                    )

            # optimize loss
            W_aug.resize((len(W_aug)))
            W_aug = loss.optimize(W_aug, optim= self.optim, is_plot = is_plot)

        # changing global parameters
        self.__B, *self__W = W_aug 
        self.__is_trained = True

        # print(self.__B)
        # print(self.__W)

        return


    def test (self, X_test : np.ndarray, Y_test : np.ndarray, is_plot : bool = False) -> float: 
        pass

    def __call__ (self, X : np.ndarray, is_plot : bool = False) -> np.ndarray : 
        assert (self.__is_trained) 
        assert (isinstance(self.__W, np.ndarray)), "W not defined"
        return X @ self.__W.reshape((len(self.__W),1)) + self.__B
