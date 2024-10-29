import numpy as np
from utils.base import Function, Optim, Algo
import matplotlib.pyplot as plt 


class LinearRegression (Algo) :
    def __init__(self, optim : Optim, *args, **kwargs) -> None:
        self.optim = optim 
        self.__is_trained : bool = False 

        self.__W  : np.ndarray | None = None
        self.__B  : np.ndarray | None = None

        self.__Y_cap_func = lambda x_aug, w_aug : x_aug @ w_aug
        self.__Error_func = lambda x_aug, y, w_aug :  self.__Y_cap_func(x_aug,w_aug) - y
        self.__model : Function | None = None
        return

    def _init_params (self, size : tuple) -> None :
        if (not self.__is_trained) : 
            self.__W = np.random.normal(size = size[1]) 
            self.__B = np.random.normal(size = 1)

    def _reset (self) -> None : 
        if (self.__is_trained) : 
            self.__W = None
            self.__B = None
            self.__model = None
            self.__is_trained = False 

    def __call__ (self, X : np.ndarray, is_plot : bool = False) -> np.ndarray : 
        assert (isinstance(self.__model, Function) and isinstance(self.__W, np.ndarray)), "Model not trained" 
        assert (self.__W.shape[0] == X.shape[1]), f"Input dimension doesn't match\nWeights : {self.__W.shape} Input : {X.shape}"

        return self.__model(X)


    def train (self, X_train : np.ndarray, Y_train : np.ndarray, epochs : int = 1, is_plot : bool = False) -> None : 
        assert (X_train.shape[0] == Y_train.shape[0]), f"Dimension mismatch X : {X_train.shape} and Y : {Y_train.shape}"

        # if model already trained, reset the parameters
        if (self.__is_trained) : 
            self._reset()

        # initialise random parameters 
        self._init_params(X_train.shape) # W -> (D,) and B -> (1,)

        assert (isinstance(self.__W,np.ndarray)), "Error in initialising W"
        assert (isinstance(self.__B,np.ndarray)), "Error in initialising B"

        # construction of Augmented matrices
        one_coloums = np.ones((X_train.shape[0], 1))
        X_train_aug = np.concatenate([one_coloums, X_train], axis = 1)  # X -> (N, D+1)
        W_aug = np.concatenate([self.__B, self.__W], axis = 0) # W -> (D+1, )

        # training loop
        for epoch in range(epochs)  : 

            # initialising Loss Function 
            loss = Function(
                    func = lambda bw : 1/(2*Y_train.shape[0])  * np.square(np.linalg.norm(self.__Error_func(X_train_aug, Y_train, bw))), 
                    grad_func = lambda bw : 1/(Y_train.shape[0]) * X_train_aug.T @ self.__Error_func(X_train_aug, Y_train, bw),
                    hessian_func= lambda bw : 1/(Y_train.shape[0]) * X_train_aug.T  @ X_train_aug,
                    name = "mean_square_error"
                    )

            # scatter plotting 
            if (is_plot) :
                if (X_train.shape[1] == 1):
                    plt.scatter(X_train, Y_train)
                    plt.show()

                if (X_train.shape[1] == 2):
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap= "cool")
                    plt.colorbar(scatter, label='Y')
                    plt.show()

            # # optimize loss
            if (is_plot and W_aug.shape[0] > 2) : 
                print(f"Can't plot Loss Function, Augmented Weight Matrix shape : {W_aug.shape[0]}") 
                W_aug = loss.optimize(W_aug, optim= self.optim)
            else : 
                loss.plot()
                W_aug = loss.optimize(W_aug, optim= self.optim, is_plot = is_plot)

        # changing global parameters
        self.__is_trained = True

        # weights and bias
        W_split = np.split(W_aug,[1,W_aug.shape[0]])
        self.__B = W_split[0]
        self.__W = W_split[1]

        # model function d
        self.__model = Function(
                func = lambda x : x @ self.__W + self.__B
                )

        return

    def test (self, X_test : np.ndarray, Y_test : np.ndarray, is_plot : bool = False) -> np.float32 : 
        y_cap = self.__call__(X_test)
        error = y_cap  - Y_test
        return np.linalg.norm(error)





class LogisticRegression (Algo) :
    def __init__(self, optim : Optim, *args, **kwargs) -> None:
        self.optim = optim 
        self.__is_trained : bool = False 

        self.__W  : np.ndarray | None = None
        self.__B  : np.ndarray | None = None

        self.__Y_cap_func = lambda x_aug, w_aug : 1 / (1 + np.exp( - x_aug @ w_aug) ) 
        self.__Error_func = lambda x_aug, y, w_aug :  self.__Y_cap_func(x_aug,w_aug) - y
        self.__model : Function | None = None
        return

    def _init_params (self, size : tuple) -> None :
        if (not self.__is_trained) : 
            self.__W = np.random.normal(size = size[1]) 
            self.__B = np.random.normal(size = 1)

    def _reset (self) -> None : 
        if (self.__is_trained) : 
            self.__W = None
            self.__B = None
            self.__model = None
            self.__is_trained = False 

    def __call__ (self, X : np.ndarray, is_plot : bool = False) -> np.ndarray : 
        assert (isinstance(self.__model, Function) and isinstance(self.__W, np.ndarray)), "Model not trained" 
        assert (self.__W.shape[0] == X.shape[1]), f"Input dimension doesn't match\nWeights : {self.__W.shape} Input : {X.shape}"

        return self.__model(X)


    def train (self, X_train : np.ndarray, Y_train : np.ndarray, epochs : int = 1, is_plot : bool = False) -> None : 
        assert (X_train.shape[0] == Y_train.shape[0]), f"Dimension mismatch X : {X_train.shape} and Y : {Y_train.shape}"

        # if model already trained, reset the parameters
        if (self.__is_trained) : 
            self._reset()

        # initialise random parameters 
        self._init_params(X_train.shape) # W -> (D,) and B -> (1,)

        assert (isinstance(self.__W,np.ndarray)), "Error in initialising W"
        assert (isinstance(self.__B,np.ndarray)), "Error in initialising B"

        # construction of Augmented matrices
        one_coloums = np.ones((X_train.shape[0], 1))
        X_train_aug = np.concatenate([one_coloums, X_train], axis = 1)  # X -> (N, D+1)
        W_aug = np.concatenate([self.__B, self.__W], axis = 0) # W -> (D+1, )

        # training loop
        for epoch in range(epochs)  : 

            # initialising Loss Function 
            loss = Function(
                    func = lambda bw : - 1/(Y_train.shape[0])  * np.sum (Y_train * np.log(self.__Y_cap_func(X_train_aug, bw)) + (1-Y_train) * np.log(1-self.__Y_cap_func(X_train_aug, bw))), 
                    grad_func = lambda bw : 1/(Y_train.shape[0]) * X_train_aug.T @ self.__Error_func(X_train_aug, Y_train, bw),
                    hessian_func= lambda bw : 1/(Y_train.shape[0]) * X_train_aug.T @ ((self.__Y_cap_func(X_train_aug, bw) * (1 - self.__Y_cap_func(X_train_aug, bw)) ).reshape((Y_train.shape[0],1)) * X_train_aug),
                    name = "cross_entropy_loss"
                    )


            # scatter plotting 
            if (is_plot) :
                if (X_train.shape[1] == 1):
                    plt.scatter(X_train, Y_train)
                    plt.show()

                if (X_train.shape[1] == 2):
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap= "cool")
                    plt.colorbar(scatter, label='Y')
                    plt.show()


            # # optimize loss
            if (is_plot and W_aug.shape[0] > 2) : 
                print(f"Can't plot Loss Function, Augmented Weight Matrix shape : {W_aug.shape[0]}") 
                W_aug = loss.optimize(W_aug, optim= self.optim)
            else : 
                loss.plot()
                W_aug = loss.optimize(W_aug, optim= self.optim, is_plot = is_plot)

        # changing global parameters
        self.__is_trained = True

        # weights and bias
        W_split = np.split(W_aug,[1,W_aug.shape[0]])
        self.__B = W_split[0]
        self.__W = W_split[1]

        # model function d
        self.__model = Function(
                func = lambda x : 1 / (1 + np.exp(-x @ self.__W + self.__B))
                )

        return

    def test (self, X_test : np.ndarray, Y_test : np.ndarray, is_plot : bool = False) -> np.float32 : 
        y_cap = self.__call__(X_test)
        error = y_cap  - Y_test
        return np.linalg.norm(error)
