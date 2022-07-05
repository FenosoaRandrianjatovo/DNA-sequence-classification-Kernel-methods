import numpy as np
import cvxopt
from kernels import linear_kernel, polynomial_kernel, rbf_kernel

class Logisticregression():
    def __init__(self,train_data,train_labels,lamda=0.2,lr=0.01,decay=10,batch_size=64,epoch=10,print_every = 10):
        dummy_once = np.ones((len(train_data),1))
        self.train_data = np.hstack((dummy_once,train_data))
        self.train_labels = train_labels
        
        self.params = np.zeros((len(self.train_data[0]),1))
        
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.print_every = print_every
        self._lambda = lamda
        self.decay = decay
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def cost(self,y,y_pred):
        return -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
    
    def gradient(self,y,y_pred,x):
        return np.dot(x.T,(y_pred-y))+(2*(self._lambda/len(self.train_labels ))*self.params)
    
    def train(self):
        for i in range(self.epoch):
            for j in range(len(self.train_labels)//self.batch_size):
                idx = list(np.random.choice(np.arange(len(self.train_labels)),self.batch_size,replace=False))
                data = self.train_data[idx]
                label = self.train_labels[idx]

                y_pred = self.sigmoid(np.dot(data,self.params))
                loss = self.cost(label,y_pred)

                gra = self.gradient(label,y_pred,data)
                self.params -= self.lr*gra

                self.lr *= (1. / (1. + self.decay * i))
            
            if self.print_every:
                if i%self.print_every == 0 or i == self.epoch-1:
                    print('Epoch : {}  Loss: {}'.format(i,loss))
    def predict(self,test_data):

        result = self.sigmoid(np.dot(test_data,self.params[1:])+self.params[0])
        return np.where(result <= 0.5, 0, 1)
    
    def evaluate(self,test_data,labels):
        accuracy = self.accuracy_score(self.predict(test_data),labels)
        
        return accuracy

    def accuracy_score(self,y_pred,label):
        return np.mean(y_pred==label)


class KernelMethodBase(object):
    kernels_ = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rbf': rbf_kernel,
        
    }
    def __init__(self, kernel='linear', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf' or self.kernel_name == 'gaussian':
            params['sigma'] = kwargs.get('sigma', None)
        if self.kernel_name == 'polynomial':
            params['power'] = kwargs.get('power', None)
        return params

    def fit(self, X, y, **kwargs):
        return self
        
    def decision_function(self, X):
        pass

    def predict(self, X):
        pass



class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit(self, X, y, sample_weights=None):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        if sample_weights is not None:
            w_sqrt = np.sqrt(sample_weights)
            self.X_train = self.X_train * w_sqrt[:, None]
            self.y_train = self.y_train * w_sqrt
        
        A = self.kernel_function_(X,X,**self.kernel_parameters)
        A[np.diag_indices_from(A)] = np.add(A[np.diag_indices_from(A)],n*self.lambd)
        self.alpha = np.linalg.solve(A , self.y_train)

        return self
    
    def decision_function(self, X):
        K_x = self.kernel_function_(X,self.X_train, **self.kernel_parameters)
        return K_x@self.alpha
    
    def predict(self, X):
        return self.decision_function(X)





class KernelSVM(KernelMethodBase):
    def __init__(self, C=0.1, **kwargs):
        self.C = C
        super(KernelSVM, self).__init__(**kwargs)
        
    def cvxopt_qp(self,P, q, G, h, A, b):
        P = .5 * (P + P.T)
        cvx_matrices = [
            cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
        ]
        #cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
        return np.array(solution['x']).flatten()
    
    def svm_dual_soft_to_qp_kernel(self,K, y, C=1):
        n = K.shape[0]
        assert (len(y) == n)


        # P = np.diag(y) @ K @ np.diag(y)
        P = np.diag(y).dot(K).dot(np.diag(y))
        eps = 1e-12
        P += eps * np.eye(n)
        q = - np.ones(n)
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.hstack([np.zeros(n), C * np.ones(n)])
        A = y[np.newaxis, :]
        b = np.array([0.])
        return P, q, G, h, A.astype(float), b


    def fit(self, X, y, tol=1e-8):
        n, _ = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        print("KernelSVM Fitting ")
        print(" ")
        K = self.kernel_function_(
            self.X_train, self.X_train, **self.kernel_parameters)
        

        self.alpha = self.cvxopt_qp(*self.svm_dual_soft_to_qp_kernel(K, y, C=self.C))
        

        sv = np.logical_and((self.alpha > tol), (self.C - self.alpha > tol))
        self.bias = y[sv] - K[sv].dot(self.alpha * y)
        self.bias = self.bias.mean()

        self.support_vector_indices = np.nonzero(sv)[0]
        print("KernelSVM Fitting DONE ")
        print(" ")
        return self
        
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)
        return K_x @ (self.alpha * self.y_train) + self.bias

    def predict(self, X):
        print("KernelSVM Predicting ")
        print(" ")
        return np.sign(self.decision_function(X))