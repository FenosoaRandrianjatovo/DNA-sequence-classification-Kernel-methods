
from models import Logisticregression, KernelRidgeRegression, KernelSVM
from utils import change_label, embedded_data, train_test_split, Saved
import cross_validation as cv
from parser import KERNEL, CC, LAMBDA,SIGMA, KMER_SIZE, POWER, KK


if __name__ == "__main__":


    
    kmer_size  =  10 if KMER_SIZE   == None else KMER_SIZE
    kernel     = 'linear' if KERNEL == None else KERNEL
    power      =  1 if POWER == None else POWER
    sigma      =  144.63369545821408 if SIGMA == None else SIGMA
    lambd      =  57.6206097294714 if LAMBDA == None else LAMBDA
    k          =  4 if KK == None else KK
    C          =  1.4035505083702005 if CC == None else CC
    

    print("*"*20,"Loading the Data","*"*20)
    print( )
    X=embedded_data(kmer_size)[:2000,:]
    y=change_label(label=1)

    print("Splitiing the Data for Cross validation")

    print(" \n")

    X_train, X_test, y_train, y_test  = train_test_split(X,y,p=0.01)

    print("*"*20,"Fitting the model","*"*20)
    print( )
    model = KernelSVM(C=C, 
                      kernel=kernel, 
                      sigma=sigma,
                      lambd=lambd, 
                      power=power)

    model.fit(X_train, y_train.flatten())
    print("*"*20,"Make prediction ","*"*20)
    print( )
    y_pred  = model.predict(embedded_data(kmer_size)[2000:,:])
    
    print("Searching for the overall accuracy ")
    print("\n")
    accuracy = cv.CrossValidatation(X,y,
                                    model_name='KernelSVM',
                                    kernel = kernel,
                                    k=k,
                                    power = power,
                                    sigma = sigma,
                                    C = C,
                                    lambd = lambd)
    print("DONE")
    print("\n")
    print("*"*20,"Saving into csv file foramt","*"*20)

    Saved(y_pred, name="Your_prediction_Test")
    print(" ")
    
    print(f"{accuracy} is th accuracy for this Hyperparameter after K-fold cross validation")


